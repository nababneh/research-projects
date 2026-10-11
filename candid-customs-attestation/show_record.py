"""Print one attestation record written by the reference node, then verify it.

The output is reproduced verbatim as a listing in the manuscript.
"""
import struct
import reference_impl as R

node = R.Node(n_endorsers=3, block_size=2)
decl = "D2021061400417"
declared = node.propose(decl, "value", "KRW 1,250,000", "declarant")
attested = node.propose(decl, "value", "KRW 1,250,000", "partner_export_record")
v, inv = node.validate_and_commit([declared, attested])
row = node.ledger.db.execute(
    "SELECT k, ver, commitment, src FROM state WHERE k = ?", (attested["key"],)).fetchone()
print(f"$ python3 show_record.py")
print(f"block {node.ledger.height} committed: {v} valid, {inv} invalid")
print(f"key        {row[0]}")
print(f"version    {row[1]}")
print(f"source     {row[3]}")
print(f"commitment {row[2].hex()}")
print("signatures " + ", ".join(f"peer{i}: {len(s)} B" for i, s in enumerate(attested['sigs'])))
same = node.ledger.db.execute("SELECT commitment FROM state WHERE k = ?",
                              (declared["key"],)).fetchone()[0] == row[2]
print(f"declared commitment equals source commitment: {same}  ->  a = {1 if same else 0}")
salt = node.admin.salt(decl, "value")
print(f"disclose to partner: value 'KRW 1,250,000', salt {salt.hex()}")
print(f"partner check SHA256(salt || value) == commitment: "
      f"{node.admin.open(decl, 'value', 'KRW 1,250,000', row[2])}")
print(f"tampered value 'KRW 1,050,000' accepted: "
      f"{node.admin.open(decl, 'value', 'KRW 1,050,000', row[2])}")
ok = all(p.verify(attested['payload'], s) for p, s in zip(node.peers, attested['sigs']))
print(f"endorsement signatures verify: {ok}")
