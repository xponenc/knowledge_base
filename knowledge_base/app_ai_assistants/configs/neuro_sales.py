NS_ASSISTANT_CONFIG = {
  "name": "Sales Assistant",
  "description": "Ассистент для продаж",
  "type": "neuro_sales",
  "blocks": [
    {
      "name": "greeting",
      "block_type": "extractor"
    },
    {
      "name": "extractors",
      "block_type": "parallel",
      "children": [
        {"name": "needs", "block_type": "extractor"},
        {"name": "benefits", "block_type": "extractor"},
        {"name": "objections", "block_type": "extractor"},
        {"name": "resolved_objections", "block_type": "extractor"},
        {"name": "tariffs", "block_type": "extractor"},
        {"name": "topic_phrases", "block_type": "extractor"}
      ]
    },
    {
      "name": "parallel_search_and_router",
      "block_type": "parallel",
      "children": [
        {
          "name": "summary_and_router",
          "block_type": "sequential",
          "children": [
            {"name": "summary", "block_type": "summary"},
            {"name": "expert_router", "block_type": "router"}
          ]
        },
        {"name": "search_index", "block_type": "retriever"},
        {"name": "original_inputs", "block_type": "passthrough"}
      ]
    },
    {
      "name": "experts",
      "block_type": "parallel",
      "children": [
        {"name": "Специалист по отработке возражений", "block_type": "expert"},
        {"name": "Специалист по презентациям", "block_type": "expert"},
        {"name": "Специалист по Zoom", "block_type": "expert"},
        {"name": "Специалист по выявлению потребностей", "block_type": "expert"},
        {"name": "Специалист по завершению", "block_type": "expert"}
      ]
    },
    {
      "name": "senior",
      "block_type": "senior"
    },
    {
      "name": "stylist",
      "block_type": "stylist"
    },
    {
      "name": "remove_greeting",
      "block_type": "extractor"
    }
  ]
}