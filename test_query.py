from rag_chain import build_chain

chain = build_chain()
q = "Quelle est la première ligne de mon document ?"
res = chain(q)
print("Réponse :", res["result"])
print("\n=== Passages récupérés ===")
for i, d in enumerate(res["source_documents"], 1):
    print(f"{i}.", d.page_content[:200].replace("\n", " "))
