from src.sentiment import predict_sentiment, analyze_news_batch, aggregate_sentiment

texts = [
    "Apple reports stronger than expected earnings and raises guidance.",
    "Tesla shares fall after weak delivery numbers.",
    "Microsoft announces a regular board meeting next week."
]

print("Single prediction:")
print(predict_sentiment(texts[0]))

print("\nBatch predictions:")
sentiment_df = analyze_news_batch(texts)
print(sentiment_df)

print("\nAggregate prediction:")
summary = aggregate_sentiment(sentiment_df)
print(summary)