import sys, glob
from rag_chain import ingest

paths = sys.argv[1:] or glob.glob("data/*")
if not paths:
    print("⚠️  Aucun fichier à ingérer")
    sys.exit(1)

ingest(paths)
print("✅  Ingestion terminée")
