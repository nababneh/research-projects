"""
Reference implementation of the CANDID attestation plane, and a benchmark.

This is a working implementation, not a simulation. It performs real
cryptography and real state-database work:

  commitment      SHA-256 over the field-group value with a per-field salt
                  derived by HKDF from an administration-held key
  endorsement     ECDSA over NIST P-256, one signature per endorsing peer
  validation      signature verification against the endorsement policy,
                  followed by an MVCC read-write-set conflict check
  commit          batched write to a SQLite state database with the ledger
                  block appended

The pipeline follows execute-order-validate. Blocks are cut on a size or
timeout criterion. Everything reported by `benchmark` is measured on the host
that runs it; nothing is calibrated to a published figure.

This is a reference node, not Hyperledger Fabric. It reproduces the critical
path that determines CANDID's cost, and the measured per-operation costs can be
compared against published Fabric measurements, but a Fabric deployment adds
gossip, MSP and chaincode-container overheads that this does not model.
"""
import hashlib, hmac, json, os, sqlite3, statistics, struct, time
from concurrent.futures import ThreadPoolExecutor

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils
from cryptography.exceptions import InvalidSignature

FIELDS = ("value", "classification", "origin")


def hkdf(key: bytes, info: bytes, n: int = 16) -> bytes:
    prk = hmac.new(b"candid-salt", key, hashlib.sha256).digest()
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()[:n]


class Administration:
    """Holds the salt key and can open any commitment it has written."""

    def __init__(self, key: bytes = None):
        self.key = key or os.urandom(32)

    def salt(self, decl_id: str, field: str) -> bytes:
        return hkdf(self.key, f"{decl_id}|{field}".encode())

    def commit(self, decl_id: str, field: str, value: str) -> bytes:
        s = self.salt(decl_id, field)
        return hashlib.sha256(s + value.encode()).digest()

    def open(self, decl_id: str, field: str, value: str, commitment: bytes) -> bool:
        return hmac.compare_digest(self.commit(decl_id, field, value), commitment)


class Peer:
    def __init__(self, name):
        self.name = name
        self.sk = ec.generate_private_key(ec.SECP256R1())
        self.pk = self.sk.public_key()

    def endorse(self, payload: bytes) -> bytes:
        return self.sk.sign(payload, ec.ECDSA(hashes.SHA256()))

    def verify(self, payload: bytes, sig: bytes) -> bool:
        try:
            self.pk.verify(sig, payload, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False


class Ledger:
    """State database and block store.

    durable=False keeps the database in memory with journalling and fsync
    disabled. It measures the computational critical path only.
    durable=True writes a file-backed database in WAL mode with
    synchronous=FULL, which forces an fsync at every block commit.
    """

    def __init__(self, path=":memory:", durable=False):
        self.db = sqlite3.connect(path, check_same_thread=False)
        if durable:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
        else:
            self.db.execute("PRAGMA journal_mode=MEMORY")
            self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("CREATE TABLE IF NOT EXISTS state("
                        "k TEXT PRIMARY KEY, ver INTEGER, commitment BLOB, src TEXT, ts REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS blocks("
                        "num INTEGER PRIMARY KEY, ntx INTEGER, hdr BLOB)")
        self.height = 0
        self.prev = b"\x00" * 32

    def versions(self, keys):
        q = ",".join("?" * len(keys))
        cur = self.db.execute(f"SELECT k, ver FROM state WHERE k IN ({q})", keys)
        return dict(cur.fetchall())

    def commit_block(self, txs):
        """MVCC check then batched write. Returns (valid, invalid) counts."""
        keys = [t["key"] for t in txs]
        cur = self.versions(keys)
        seen, rows, valid, invalid = {}, [], 0, 0
        for t in txs:
            k = t["key"]
            live = seen.get(k, cur.get(k, 0))
            if t["read_ver"] != live:
                invalid += 1
                continue
            seen[k] = live + 1
            rows.append((k, live + 1, t["commitment"], t["src"], t["ts"]))
            valid += 1
        self.db.executemany(
            "INSERT INTO state(k,ver,commitment,src,ts) VALUES(?,?,?,?,?) "
            "ON CONFLICT(k) DO UPDATE SET ver=excluded.ver, commitment=excluded.commitment,"
            " src=excluded.src, ts=excluded.ts", rows)
        self.height += 1
        hdr = hashlib.sha256(self.prev + struct.pack("<II", self.height, len(txs))
                             + b"".join(t["commitment"] for t in txs)).digest()
        self.prev = hdr
        self.db.execute("INSERT INTO blocks(num,ntx,hdr) VALUES(?,?,?)",
                        (self.height, len(txs), hdr))
        self.db.commit()
        return valid, invalid


class Node:
    def __init__(self, n_endorsers=3, block_size=250, block_timeout=0.5,
                 verify_workers=4, path=":memory:", durable=False):
        self.admin = Administration()
        self.peers = [Peer(f"peer{i}") for i in range(n_endorsers)]
        if durable and path == ":memory:":
            import tempfile
            path = os.path.join(tempfile.mkdtemp(prefix="candid_"), "ledger.db")
        self.ledger = Ledger(path, durable=durable)
        self.block_size = block_size
        self.block_timeout = block_timeout
        self.pool = ThreadPoolExecutor(max_workers=verify_workers)

    # ---- execute ----
    def propose(self, decl_id, field, value, src):
        c = self.admin.commit(decl_id, field, value)
        key = f"{decl_id}|{field}|{src}"
        rv = self.ledger.versions([key]).get(key, 0)
        payload = key.encode() + c + struct.pack("<I", rv)
        sigs = [p.endorse(payload) for p in self.peers]
        return dict(key=key, commitment=c, src=src, read_ver=rv,
                    payload=payload, sigs=sigs, ts=time.perf_counter())

    # ---- validate ----
    def _verify_one(self, tx):
        return all(p.verify(tx["payload"], s) for p, s in zip(self.peers, tx["sigs"]))

    def validate_and_commit(self, txs):
        oks = list(self.pool.map(self._verify_one, txs))
        good = [t for t, o in zip(txs, oks) if o]
        return self.ledger.commit_block(good)


def stage_costs(n=2000):
    """Measure each stage in isolation."""
    node = Node()
    admin, peers = node.admin, node.peers
    vals = [f"KRW {1000+i}" for i in range(n)]
    t = time.perf_counter()
    cs = [admin.commit(f"D{i}", "value", vals[i]) for i in range(n)]
    t_commit = (time.perf_counter() - t) / n
    payload = b"k" * 48 + cs[0] + b"\x00\x00\x00\x00"
    t = time.perf_counter()
    sigs = [peers[0].endorse(payload) for _ in range(n)]
    t_sign = (time.perf_counter() - t) / n
    t = time.perf_counter()
    for s in sigs: peers[0].verify(payload, s)
    t_verify = (time.perf_counter() - t) / n
    return dict(commitment_us=t_commit * 1e6, sign_us=t_sign * 1e6,
                verify_us=t_verify * 1e6)


def benchmark(total=20000, block_size=250, n_endorsers=3, verify_workers=4,
              batch=500, durable=False):
    """Closed loop: proposals are generated back to back and a block is cut
    whenever block_size transactions are buffered. Throughput is committed
    transactions over wall time. Latency runs from proposal to block commit.
    No transaction is queued outside the buffer and none is rejected."""
    node = Node(n_endorsers=n_endorsers, block_size=block_size,
                verify_workers=verify_workers, durable=durable)
    lat, done = [], 0
    t0 = time.perf_counter()
    buf = []
    for i in range(total):
        f = FIELDS[i % 3]
        tx = node.propose(f"D{i//3:07d}", f, f"v{i}", f"src{i%6}")
        buf.append(tx)
        if len(buf) >= block_size:
            node.validate_and_commit(buf)
            now = time.perf_counter()
            lat.extend(now - t["ts"] for t in buf)
            done += len(buf); buf = []
    if buf:
        node.validate_and_commit(buf)
        now = time.perf_counter()
        lat.extend(now - t["ts"] for t in buf); done += len(buf)
    el = time.perf_counter() - t0
    lat = sorted(lat)
    return dict(committed=done, seconds=round(el, 3),
                tps=round(done / el, 1),
                mean_latency_ms=round(1000 * sum(lat) / len(lat), 2),
                p50_ms=round(1000 * lat[len(lat)//2], 2),
                p95_ms=round(1000 * lat[int(len(lat)*.95)], 2),
                p99_ms=round(1000 * lat[int(len(lat)*.99)], 2),
                block_height=node.ledger.height)


def load_sweep(rate, seconds=6.0, block_size=250, n_endorsers=3, verify_workers=4,
               block_timeout=0.5, seed=0, durable=False):
    """Open loop with an explicit backlog.

    Poisson arrival times are drawn in advance for the whole window. The node
    admits every transaction whose arrival time has passed, endorses it and
    buffers it. A block is cut when block_size transactions are buffered or
    when the oldest buffered transaction has waited block_timeout seconds.
    Latency runs from the scheduled arrival time to block commit, which means
    queueing delay is included. Arrivals are never dropped. After the window
    closes the backlog is drained, and the backlog size at the close is
    reported.
    """
    import random
    rnd = random.Random(seed)
    arrivals, t = [], 0.0
    while True:
        t += rnd.expovariate(rate)
        if t > seconds:
            break
        arrivals.append(t)
    node = Node(n_endorsers=n_endorsers, block_size=block_size,
                verify_workers=verify_workers, durable=durable)
    t0 = time.perf_counter()
    buf, lat = [], []
    i, n = 0, len(arrivals)
    backlog_at_close = None
    last_commit = t0
    while i < n or buf:
        now = time.perf_counter() - t0
        if backlog_at_close is None and now >= seconds:
            backlog_at_close = (n - i) + len(buf)
        # admit everything that has arrived, up to one block at a time
        while i < n and arrivals[i] <= now and len(buf) < block_size:
            f = FIELDS[i % 3]
            tx = node.propose(f"D{i//3:07d}", f, f"v{i}", f"src{i%6}")
            tx["arr"] = arrivals[i]
            buf.append(tx); i += 1
        now = time.perf_counter() - t0
        full = len(buf) >= block_size
        stale = buf and (now - buf[0]["arr"] >= block_timeout)
        draining = i >= n and buf
        if full or stale or draining:
            node.validate_and_commit(buf)
            fin = time.perf_counter() - t0
            lat.extend(fin - tx["arr"] for tx in buf)
            buf = []; last_commit = fin
        elif i < n and arrivals[i] > now:
            time.sleep(min(0.0005, arrivals[i] - now))
    if backlog_at_close is None:
        backlog_at_close = 0
    lat = sorted(lat) or [0.0]
    q = lambda p: 1000 * lat[min(int(len(lat) * p), len(lat) - 1)]
    return dict(offered_tps=rate, arrivals=n,
                achieved_tps=round(n / max(last_commit, seconds), 1),
                mean_latency_ms=round(1000 * sum(lat) / len(lat), 2),
                p50_latency_ms=round(q(.50), 2), p95_latency_ms=round(q(.95), 2),
                p99_latency_ms=round(q(.99), 2),
                backlog_at_close=backlog_at_close, drain_s=round(max(last_commit - seconds, 0), 3),
                committed=n)


def footprint(n=2000, n_endorsers=3, attest_per_decl=4.65, document_bytes=260_000,
              decl_csv=None):
    """Measure the serialised size of an on-chain attestation record.

    record     key, commitment, endorsement signatures (DER), read version
               (4 bytes) and timestamp (8 bytes), as serialised by this node
    framing    the transaction envelope a permissioned ledger adds. We measure
               a DER-encoded X.509 creator certificate on P-256 and add a
               32-byte transaction identifier, a 32-byte channel identifier
               and one client signature.
    baselines  a bare declaration record is the JSON serialisation of a
               substrate row plus the same framing. The document-commitment
               design adds document_bytes per declaration. That figure is an
               assumption (four supporting documents of about 65 kB each) and
               not a measurement.
    """
    import datetime
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    node = Node(n_endorsers=n_endorsers)
    sizes, sig_sizes = [], []
    for i in range(n):
        tx = node.propose(f"D{i:07d}", FIELDS[i % 3], f"v{i}", f"src{i%6}")
        sig = sum(len(s) for s in tx["sigs"])
        rec = len(tx["key"].encode()) + len(tx["commitment"]) + sig + 4 + 8
        sizes.append(rec); sig_sizes.append(sig)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COUNTRY_NAME, "AE"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Customs Administration"),
                      x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "client"),
                      x509.NameAttribute(NameOID.COMMON_NAME, "attestation-gateway-01")])
    now = datetime.datetime(2026, 1, 1)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=365))
            .sign(key, hashes.SHA256()))
    cert_bytes = len(cert.public_bytes(serialization.Encoding.DER))
    client_sig = len(key.sign(b"x" * 64, ec.ECDSA(hashes.SHA256())))
    framing = cert_bytes + 32 + 32 + client_sig
    rec = statistics.mean(sizes)
    out = dict(record_bytes=round(rec, 1), signature_set_bytes=round(statistics.mean(sig_sizes), 1),
               creator_certificate_bytes=cert_bytes, framing_bytes=framing,
               record_with_framing=round(rec + framing, 1),
               attestations_per_declaration=attest_per_decl,
               per_declaration_candid=round(attest_per_decl * (rec + framing), 1))
    if decl_csv and os.path.exists(decl_csv):
        import csv as _csv
        with open(decl_csv, newline="") as f:
            rows = [r for _, r in zip(range(5000), _csv.DictReader(f))]
        js = statistics.mean(len(json.dumps(r, separators=(",", ":")).encode()) for r in rows)
        out["declaration_json_bytes"] = round(js, 1)
        out["declaration_record_with_framing"] = round(js + framing, 1)
        out["document_design_per_declaration"] = round(js + framing + document_bytes, 1)
        out["assumed_document_bytes"] = document_bytes
    return out


def environment():
    import platform, sys
    import cryptography
    cpu = platform.processor()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return dict(cpu=cpu, logical_cores=os.cpu_count(), os=platform.platform(),
                python=sys.version.split()[0], cryptography=cryptography.__version__,
                sqlite=sqlite3.sqlite_version)


if __name__ == "__main__":
    import csv
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    env = environment()
    print("host:", env)
    with open(os.path.join(out, "impl_environment.json"), "w") as f:
        json.dump(env, f, indent=1)

    # stage costs, five repetitions
    reps = [stage_costs() for _ in range(5)]
    sc = {k: statistics.mean(r[k] for r in reps) for k in reps[0]}
    print("\nstage costs (microseconds):")
    for k, v in sc.items(): print(f"  {k:16s} {v:8.2f}")
    with open(os.path.join(out, "impl_stage_costs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stage", "microseconds", "sd"])
        for k in sc: w.writerow([k, round(sc[k], 3), round(statistics.stdev(r[k] for r in reps), 3)])

    # correctness: commitment hiding and opening
    a = Administration()
    c = a.commit("D1", "value", "KRW 1000")
    assert a.open("D1", "value", "KRW 1000", c)
    assert not a.open("D1", "value", "KRW 999", c)
    assert a.commit("D1", "value", "x") != a.commit("D2", "value", "x")
    print("\ncommitment binding and hiding: OK")

    # closed-loop throughput, five repetitions per configuration
    rows = []
    print("\nclosed-loop throughput by configuration:")
    for name, bs, ne, vw, dur in [("small blocks, 3 endorsers", 10, 3, 2, False),
                                  ("large blocks, 3 endorsers", 250, 3, 4, False),
                                  ("large blocks, 1 endorser", 250, 1, 4, False),
                                  ("large blocks, 5 endorsers", 250, 5, 4, False),
                                  ("large blocks, 3 endorsers, durable", 250, 3, 4, True),
                                  ("small blocks, 3 endorsers, durable", 10, 3, 2, True)]:
        for rep in range(5):
            r = benchmark(total=12000, block_size=bs, n_endorsers=ne, verify_workers=vw,
                          durable=dur)
            r.update(config=name, block_size=bs, endorsers=ne, durable=dur, rep=rep)
            rows.append(r)
        sub = [x for x in rows if x["config"] == name]
        print(f"  {name:36s} {statistics.mean(x['tps'] for x in sub):8.1f} "
              f"+- {statistics.stdev(x['tps'] for x in sub):6.1f} tx/s   "
              f"p95 {statistics.mean(x['p95_ms'] for x in sub):7.1f} ms")
    with open(os.path.join(out, "impl_benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\nopen-loop latency against offered load (3 endorsers, block 250):")
    sweep = []
    for dur in (False, True):
        for r in [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000]:
            for rep in range(3):
                s_ = load_sweep(r, seconds=6.0, seed=rep, durable=dur)
                s_.update(durable=dur, rep=rep)
                sweep.append(s_)
            sub = [x for x in sweep if x["offered_tps"] == r and x["durable"] == dur]
            print(f"  durable={dur!s:5s} offered {r:5d}  achieved "
                  f"{statistics.mean(x['achieved_tps'] for x in sub):8.1f}  "
                  f"p95 {statistics.mean(x['p95_latency_ms'] for x in sub):9.1f} ms  "
                  f"backlog {statistics.mean(x['backlog_at_close'] for x in sub):7.0f}")
    with open(os.path.join(out, "impl_load_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)

    data = os.environ.get("KCS_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kcs_data"))
    fp = footprint(decl_csv=os.path.join(data, "df_syn_eng.csv"))
    print("\nfootprint:", fp)
    with open(os.path.join(out, "impl_footprint.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "bytes"])
        for k, v in fp.items(): w.writerow([k, v])
