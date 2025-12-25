# Sample Text Data

Text data for NLP notebooks.

## Files

| File | Rows | Task |
|------|------|------|
| product_reviews.csv | 500 | Sentiment analysis |
| qa_dataset.json | 100 | Question answering |
| articles.txt | 50 | Text generation |

## Load Data

```python
import pandas as pd
reviews = pd.read_csv('product_reviews.csv')
```
