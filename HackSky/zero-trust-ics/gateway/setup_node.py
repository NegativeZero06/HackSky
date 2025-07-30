from db import init_db, register_node
from pqc_sim import generate_key_pair

node_id = "NODE-001"
public_key, private_key = generate_key_pair()

print("Registering node:", node_id)
print("Private key (store securely):", private_key)

init_db()
register_node(node_id, private_key)