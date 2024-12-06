import warnings
warnings.filterwarnings('ignore')

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

#Download required resources for NLTK
nltk.download("punkt")



# Initialize T5 model and tokenizer for abstractive summarization
t5_model_name = "t5-large"  # Can be "t5-large" or others for better quality
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

def abstractive_summarization(text, max_length=50, min_length=25):

    input_text = f"summarize: {text}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = t5_model.generate(
        inputs, 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=2.0,
        num_beams=4, 
        early_stopping=True
    )
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extractive_summarization(text, sentence_count=3):

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Main function to choose between abstractive or extractive summarization
def summarize(text, mode="abstractive", **kwargs):

    if mode == "abstractive":
        # Pass only parameters relevant to abstractive summarization
        max_length = kwargs.get("max_length", 50)
        min_length = kwargs.get("min_length", 25)
        return abstractive_summarization(text, max_length=max_length, min_length=min_length)
    elif mode == "extractive":
        # Pass only parameters relevant to extractive summarization
        sentence_count = kwargs.get("sentence_count", 2)
        return extractive_summarization(text, sentence_count=sentence_count)
    else:
        raise ValueError("Mode must be either 'abstractive' or 'extractive'.")

# Example Usage
text = """
Manchester United, one of the most successful football clubs in the world, 
has won a record 20 English top-flight league titles, 
including 13 in the Premier League era under Sir Alex Ferguson.
 The club has also claimed 12 FA Cups, 6 League Cups, and a record 21 FA Community Shields. 
 Internationally, Manchester United has won 3 UEFA Champions League titles, 1 UEFA Europa League, 1 UEFA Cup Winners' Cup, and the FIFA Club World Cup.
 They were the first English club to win the European Cup in 1968. Known for their passionate fanbase, 
 United is a global brand with supporters worldwide
"""

# Toggle between abstractive or extractive summarization
mode = "extractive"  # Change to "abstractive" for abstractive summarization
summary = summarize(text, mode=mode, sentence_count=2)  # Use sentence_count for extractive summarization
print(f"Mode: {mode.capitalize()}")
print("Summary:")
print(summary)
