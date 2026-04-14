# Library setup
import os
import re
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import time


# Total 15 main topics & keywords
main_topics = ["tax policy", "trade policy", "free-market", "civil liberties", "gun control", "death penalty", "abortion", "LGBTQ", "drug policy",  "immigration", "gender equality", "bioethics", "nationalism", "multiculturalism", "climate change"]
keywords = [
    ["tax policy", "income tax", "tax fairness", "tax burden", "tax rate", "tax return", "tax bracket", "tax reform", "flat tax", "progressive tax"],
    ["trade policy", "global trade", "foreign trade", "international trade", "free trade", "trade barriers", "trade surplus", "trade deficit", "trade agreement", "trade sanctions"], 
    ["free market", "deregulation", "market freedom", "government regulation", "market intervention", "corporate ethics", "environmental regulation", "market concentration", "corporate lobbying", "laissez-faire"], 
    ["civil liberties", "mass surveillance", "surveillance state", "electronic surveillance", "personal freedom", "privacy rights", "constitutional rights", "individual autonomy", "free speech", "due process"],
    ["gun control", "gun laws", "gun policy", "firearm policy", "firearm laws", "gun legislation", "gun reform", "assault weapons", "gun licensing", "gun violence"],
    ["death penalty", "capital punishment", "life sentence", "maximum penalty", "justice system", "crime prevention", "victim rights", "judicial error", "wrongful conviction",  "moral opposition"],
    ["abortion rights", "pro choice", "pro life", "reproductive rights", "bodily autonomy", "abortion ban", "abortion legality", "abortion access", "abortion law", "fetal rights"], 
    ["LGBTQ rights", "same sex marriage", "gender identity", "sexual orientation", "transgender rights", "sexual diversity", "gay parenting", "LGBT inclusion", "queer rights", "marriage equality"],
    ["marijuana legalization", "medical marijuana", "cannabis policy", "drug reform", "drug laws", "cannabis use", "drug possession", "war on drugs", "drug sentencing", "recreational marijuana"],
    ["immigration policy", "border control", "immigration reform", "refugee status", "asylum law", "legal immigration", "undocumented immigrants", "immigration restriction", "immigrant assimilation", "immigrant workforce"], 
    ["gender equality", "gender roles", "gender norms", "sex discrimination", "gender bias", "feminist theory", "women's rights", "working mother", "maternal duty", "domestic labor"],
    ["physician-assisted suicide", "euthanasia", "death with dignity", "right to die", "end-of-life care", "medical ethics", "patient autonomy", "assisted dying", "euthanasia law", "bioethics"],
    ["nationalism", "patriotism", "national pride", "flag worship", "cultural supremacy", "exceptionalism", "jingoism", "national identity", "civic nationalism", "ethnic nationalism"],
    ["multiculturalism", "cultural diversity", "language tolerance", "cultural coexistence", "ethnic integration", "intercultural contact", "cross-cultural tension", "cultural acceptance", "racial identity"],
    ["climate change", "global warming", "carbon emissions", "greenhouse effect", "anthropogenic warming", "climate denial", "environmental alarm", "environmental protection",  "climate policy", "environmental priority"]
    ]

min_frequency = 3
min_text_length = 300
nli_threshold = 0.7
batch_size = 32
max_chunk_length = 1800  
hypothesis_template = "This text is about {}"
samples_per_topic = 3000  # Number of samples to collect per topic

# Set device based on GPU availability
device = 0 if torch.cuda.is_available() else -1

# NLI model name
model_name = "facebook/bart-large-mnli"

# Load NLI model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nli_pipeline = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=device)

# Function definitions 

# STEP 1. Check if text contains any keyword from current topic
def step1_contains_any_keyword(example, current_keywords):
    text = example["text"].lower()
    return any(keyword in text for keyword in current_keywords)

# STEP 2. Check if text length meets minimum length requirement
def step2_length_check(text):
    return len(text) >= min_text_length

# STEP 3. Check if text contains any keyword from current topic with minimum frequency requirement
def step3_frequency_check(text, current_keywords):
    text_lower = text.lower()
    for keyword in current_keywords:
        # Use regex for exact word boundary matching
        keyword_pattern = r'\b' + re.escape(keyword) + r'\b'
        if len(re.findall(keyword_pattern, text_lower)) >= min_frequency:
            return True
    return False

# Split long text into chunks to fit NLI model input limits
def split_text_into_chunks(text, max_len=max_chunk_length):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

# STEP 4. Split long text into chunks, run NLI pipeline, and return maximum NLI score
def step4_long_text_nli(text, hypothesis_label):
    chunks = split_text_into_chunks(text)
    try:
        results = nli_pipeline(
            chunks,
            candidate_labels=[hypothesis_label],
            hypothesis_template=hypothesis_template,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"NLI error (batch processing): {e}")
        return False, 0.0

    if isinstance(results, dict):
        results = [results]

    max_score = 0.0
    for result in results:
        if result["labels"][0] == hypothesis_label:
            max_score = max(max_score, result["scores"][0])
    return max_score >= nli_threshold, round(max_score, 3)

# Return approved sample data as structured dictionary
def process_approved_sample(sample, text, nli_score, current_id, current_topic, current_keywords):
    text_lower = text.lower()
    keyword_frequencies = {}
    total_keyword_count = 0

    for keyword in current_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        count = len(re.findall(pattern, text_lower))
        keyword_frequencies[keyword] = count
        total_keyword_count += count

    if not keyword_frequencies:
        max_keyword = "N/A"
        max_freq = 0
    else:
        max_keyword = max(keyword_frequencies, key=keyword_frequencies.get)
        max_freq = keyword_frequencies[max_keyword]

    return {
        "id": current_id,
        "topic": current_topic,
        "url": sample.get("url", ""),
        "text": text,
        "total_keyword_frequency": total_keyword_count,
        "max_keyword": max_keyword,
        "max_keyword_frequency": max_freq,
        "text_length": len(text),
        "nli_topic_score": nli_score
    }

# Create result directory for saving
os.makedirs("../c4_datasets", exist_ok=True)

print(f"🚀 Sequential NLI-based text collection started")
print(f"Number of topics: {len(main_topics)}")
print(f"Target samples per topic: {samples_per_topic}")
print(f"NLI threshold: {nli_threshold}")
print(f"Model: {model_name}")
print(f"Device: {'GPU-'+str(device) if device >= 0 else 'CPU'}")

start_time = time.time()
all_results = []

# Sequential data collection for each topic
for topic_idx, current_topic in enumerate(main_topics):
    current_keywords = keywords[topic_idx]
    hypothesis_label = current_topic
    
    print(f"\n{'='*80}")
    print(f"📝 Topic {topic_idx + 1}/{len(main_topics)}: {current_topic.upper()} data collection started")
    print(f"Number of keywords: {len(current_keywords)}")
    print(f"Target samples: {samples_per_topic}")
    print(f"NLI hypothesis: {hypothesis_template.format(hypothesis_label)}")
    print(f"{'='*80}")

    # Integrated 1-4 step filtering (streaming method)
    print(f"[{current_topic}] Integrated 1-4 step filtering started...")
    dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
    keyword_filtered_dataset = dataset.filter(
        lambda example: step1_contains_any_keyword(example, current_keywords)
    )

    # =============== Real-time filtering for final sample collection ===============
    nli_approved_samples = []
    processed_count = 0
    step1_passed_count = 0
    step2_passed_count = 0
    step3_passed_count = 0
    step4_passed_count = 0
    duplicate_count = 0
    
    # URL set for duplicate checking
    seen_urls = set()

    # Stream and apply all filtering one by one
    for sample in tqdm(keyword_filtered_dataset, desc=f"{current_topic} Full Pipeline Filtering"):
        processed_count += 1
        step1_passed_count += 1  # Already passed step 1 from keyword_filtered_dataset
        
        text = sample.get('text')
        url = sample.get('url', '')
        
        # Step 2: Length and URL check
        if not (text and step2_length_check(text) and url.startswith('https://')):
            continue
        step2_passed_count += 1

        # Duplicate check (based on URL)
        if url in seen_urls:
            duplicate_count += 1
            continue
        
        # Step 3: Frequency check  
        if not step3_frequency_check(text, current_keywords):
            continue
        step3_passed_count += 1

        # Step 4: NLI check
        is_relevant, score = step4_long_text_nli(text, hypothesis_label)
        if not is_relevant:
            continue
        step4_passed_count += 1

        # Save sample that meets all conditions
        seen_urls.add(url)  # Add URL to duplicate check set
        sample_data = process_approved_sample(
            sample, text, score, len(nli_approved_samples), current_topic, current_keywords
        )
        nli_approved_samples.append(sample_data)

        # Print progress every 10 samples
        if len(nli_approved_samples) % 10 == 0:
            avg_score = sum(s['nli_topic_score'] for s in nli_approved_samples) / len(nli_approved_samples)
            print(f"  ✅ {current_topic}: {len(nli_approved_samples):,}/{samples_per_topic:,} collected (Avg NLI score: {avg_score:.3f}, Duplicates removed: {duplicate_count:,})")

        # Stop when target sample count is reached
        if len(nli_approved_samples) >= samples_per_topic:
            break

    print(f"[{current_topic}] Filtering completed:")
    print(f"  Processed samples: {processed_count:,}")
    print(f"  Step 1 (keyword) passed: {step1_passed_count:,}")
    print(f"  Step 2 (length+URL) passed: {step2_passed_count:,}") 
    print(f"  Duplicates removed: {duplicate_count:,}")
    print(f"  Step 3 (frequency) passed: {step3_passed_count:,}")
    print(f"  Step 4 (NLI) passed: {step4_passed_count:,}")

    # =============== Final DataFrame creation, sorting and saving ===============
    df = pd.DataFrame(nli_approved_samples)

    if not df.empty:
        # Sort by NLI score in descending order
        df.sort_values(by="nli_topic_score", ascending=False, inplace=True)

        # Select top samples_per_topic if more were collected
        df = df.head(samples_per_topic).reset_index(drop=True)
        
        # Assign new sequential IDs starting from 0 to final selected samples
        df['id'] = range(len(df))
    else:
        print(f"⚠️ No valid samples passed all filtering for {current_topic} topic.")

    # Save file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_topic.replace(' ', '_')}_{current_time}.csv"
    filepath = os.path.join("datasets", filename)

    # Save sampling dataset as CSV
    if not df.empty:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

    # =============== Save collection result statistics ===============  
    result_stats = {
        'topic': current_topic,
        'filepath': filepath if not df.empty else None,
        'preliminary_samples': processed_count, # Samples passed steps 1-2
        'step3_passed': step3_passed_count,
        'step4_passed': step4_passed_count,
        'final_samples': len(df),
        'avg_length': df['text_length'].mean() if not df.empty else 0,
        'avg_keyword_freq': df['total_keyword_frequency'].mean() if not df.empty else 0,
        'avg_nli_score': df['nli_topic_score'].mean() if not df.empty else 0,
        'success_rate': len(df)/processed_count*100 if processed_count > 0 else 0
    }
    
    all_results.append(result_stats)
    print(f"✅ {current_topic.upper()} collection completed!")

end_time = time.time()
total_time = end_time - start_time

print(f"\n{'='*80}")
print(f"🎉 Complete data collection finished! (Time taken: {total_time/60:.1f} minutes)")
print(f"{'='*80}")

print(f"\n=== Final Collection Results ===")
total_collected = 0
for result in all_results:
    print(f"\n📊 {result['topic'].upper()}:")
    print(f"  Save location: {result['filepath'] if result['filepath'] else 'No file saved'}")
    print(f"  Processed samples: {result['preliminary_samples']:,}")
    print(f"  Step 3 passed: {result['step3_passed']:,}") 
    print(f"  Step 4 NLI passed: {result['step4_passed']:,}")
    print(f"  Final collected: {result['final_samples']:,}")
    if result['final_samples'] > 0:
        print(f"  Average text length: {result['avg_length']:.0f} chars")
        print(f"  Average keyword frequency: {result['avg_keyword_freq']:.1f} times")
        print(f"  Average NLI score: {result['avg_nli_score']:.3f}")
        print(f"  Collection success rate: {result['success_rate']:.2f}%")
    total_collected += result['final_samples']

print(f"\n=== Overall Summary ===")
print(f"Total collected samples: {total_collected:,}")
print(f"Average per topic: {total_collected/len(main_topics):.0f}")
print(f"Processed topics: {', '.join(main_topics)}")
print(f"Target per topic: {samples_per_topic:,}")
print(f"Minimum frequency threshold: {min_frequency} times")
print(f"Minimum text length: {min_text_length} chars")
print(f"NLI threshold: {nli_threshold}")
print(f"Total processing time: {total_time/60:.1f} minutes")