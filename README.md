# F1 Fan Sentiment Analyser

I built this because I wanted to answer a question I kept wondering about as an F1 fan — 
do people actually hate Verstappen or do they just hate that he keeps winning? 
And which drivers get unfair criticism compared to their actual performance?

So I built an AI that reads fan comments and detects the emotion behind them.

## What it does

You open the app and it analyses 50 real F1 fan comments using an emotion detection AI. 
It tells you which drivers get the most anger, sadness, joy, or surprise from fans — 
and which comments the AI wasn't confident enough to label at all.

There's also a live Reddit button that tries to fetch real comments from r/formula1 
right now. Sometimes Reddit lets us in, sometimes it doesn't — so the app falls back 
to the default dataset gracefully instead of crashing. I learned that making something 
reliable is just as important as making it clever.

## What I actually learned

Before this project I thought using AI meant training a model. 
I didn't know you could just download one that someone already trained and use it in 
10 lines of code. That was a revelation honestly.

I also learned that AI models are not magic. I tested two models and watched them get 
things wrong in interesting ways — "Leclerc is so overrated" got labelled POSITIVE 
with 95% confidence by the basic model. That's completely wrong. Understanding WHY 
a model fails is more useful than just knowing it works.

The emotion model I ended up using detects: joy, sadness, anger, fear, surprise, 
disgust, and neutral. It's much better than basic positive/negative — but it still 
struggles with sarcasm and subtle language. I documented these limitations because 
they directly influenced how I designed my next project.

## The confidence threshold decision

One thing I'm proud of: I added a confidence threshold of 0.7. 
If the model is less than 70% sure about an emotion, the app labels it "uncertain" 
instead of showing a potentially wrong answer. 

In real AI products you don't just show whatever the model says. 
You decide when to trust it and when not to. That felt like a grown-up engineering 
decision to make.

## Built with
- Python
- HuggingFace Transformers — j-hartmann/emotion-english-distilroberta-base
- Streamlit
- Pandas
- Plotly
- Reddit public API

## Live Demo
https://f1-sentiment-dsy8vijicir9jrnagztwqt.streamlit.app/
