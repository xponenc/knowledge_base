import requests

from langchain_core.runnables import RunnableLambda


def ensemble_retriever_search(
        api_key: str
):
    def retriever_func(inputs: dict) -> str:
        """
        Выполняет поиск документов через API по search_index.
        """
        query = inputs.get("reformulated_query").content
        retriever_type = inputs.get("search_retriever_type", "multi-chain")

        if not query:
            return ""

        # Запрос к API retriever
        if retriever_type == "multi-chain":
            url = "http://localhost:8001/api/multi-retriever/search"
        else:
            url = "http://localhost:8001/api/ensemble-retriever/search"

        resp = requests.post(
            url=url,
            json={
                "query": query,
                "top_k": 5,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        documents = data["documents"]

        output_documents = [doc.get("content") for doc in documents]

        output_documents = "\n".join(
            f"Документ {i + 1}\n{doc}"
            for i, doc in enumerate(output_documents)
        )

        return output_documents

    return RunnableLambda(retriever_func)
