import os
import pandas as pd
from datetime import datetime
import time, random
import re

# Get the list of folders in the datasets directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Global variables
personas = ['opp_left', 'opp_right', 'sup_left', 'sup_right']
results_cache = {}
query = "Environment protection"    # Default query, can be modified
input_filename = "climate_change_20250806_124733.csv"

def initialize_cache(all_model_versions):
    """Initialize cache for all model-persona combinations"""
    global results_cache
    results_cache = {}
    for model_version in all_model_versions:
        for persona in personas:
            model_persona_key = f"{model_version}_{persona}"
            results_cache[model_persona_key] = {}

def create_chatgpt_content(query, text, chatgpt_model_version_list):
    responses = {}
    role_prompts = {
        'opp_left': create_role_opposed_left_prompt(query),
        'opp_right': create_role_opposed_right_prompt(query),
        'sup_left': create_role_supportive_left_prompt(query),
        'sup_right': create_role_supportive_right_prompt(query)
    }
    content_prompt = create_content_prompt(query, text)
    
    for model_version in chatgpt_model_version_list:
        from chatgpt.chatgpt_request import ChatGPT
        print(f"Processing ChatGPT model version: {model_version}")
        
        for persona, role_prompt in role_prompts.items():
            model_persona_key = f"{model_version}_{persona}"
            print(f"Processing persona: {persona}")
            
            try:
                chatgpt = ChatGPT(model_version)
                chatgpt.add_role(role_prompt)
                response = chatgpt.run(content_prompt)
                print(f"ChatGPT ({model_persona_key}) Response: {response}")
                responses[model_persona_key] = response
                
                # Add random delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error processing ChatGPT {model_persona_key}: {e}")
                responses[model_persona_key] = create_empty_result_json()
            
    return responses

def create_claude_content(query, text, claude_model_version_list):
    responses = {}
    role_prompts = {
        'opp_left': create_role_opposed_left_prompt(query),
        'opp_right': create_role_opposed_right_prompt(query),
        'sup_left': create_role_supportive_left_prompt(query),
        'sup_right': create_role_supportive_right_prompt(query)
    }
    content_prompt = create_content_prompt(query, text)
    
    for model_version in claude_model_version_list:
        from claude.claude_request import Claude
        print(f"Processing Claude model version: {model_version}")
        
        for persona, role_prompt in role_prompts.items():
            model_persona_key = f"{model_version}_{persona}"
            print(f"Processing persona: {persona}")
            
            try:
                claude = Claude(model_version)
                claude.add_role(role_prompt)
                response = claude.run(content_prompt)
                print(f"Claude ({model_persona_key}) Response: {response}")
                responses[model_persona_key] = response
                
                # Add random delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error processing Claude {model_persona_key}: {e}")
                responses[model_persona_key] = create_empty_result_json()
            
    return responses

def create_role_opposed_left_prompt(query):
    role_prompt_path = os.path.join(current_dir, 'prompt', 'prompt_role_opp-left.txt')
    with open(role_prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template.format(topic=query)

def create_role_opposed_right_prompt(query):
    role_prompt_path = os.path.join(current_dir, 'prompt', 'prompt_role_opp-right.txt')
    with open(role_prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template.format(topic=query)

def create_role_supportive_left_prompt(query):
    role_prompt_path = os.path.join(current_dir, 'prompt', 'prompt_role_sup-left.txt')
    with open(role_prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template.format(topic=query)

def create_role_supportive_right_prompt(query):
    role_prompt_path = os.path.join(current_dir, 'prompt', 'prompt_role_sup-right.txt')
    with open(role_prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template.format(topic=query)

def create_content_prompt(query, text):
    content_prompt_path = os.path.join(current_dir, 'prompt', 'prompt_input.txt')
    with open(content_prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template.format(topic=query, text=text)

def create_empty_result_json():
    return """
    {
    "Political": {
        "label": "Undecided",
        "score": 0.0
    },
    "Stance": {
        "label": "Undecided",
        "score": 0.0
    },
    "Reasoning": "Failed to process"
    }
    """

def ensure_columns_exist(df, all_model_versions):
    """Ensure all required columns exist in the dataframe"""
    for model_version in all_model_versions:
        for persona in personas:
            model_persona_key = f"{model_version}_{persona}"
            if model_persona_key not in df.columns:
                df[model_persona_key] = ""
    return df

def get_df(claude_model_version_list, chatgpt_model_version_list, other_model_version_list, file_path, annotation_file_path):
    global query, results_cache
    
    # Combine all model versions
    all_model_versions = claude_model_version_list + chatgpt_model_version_list + other_model_version_list
    
    # Initialize cache
    initialize_cache(all_model_versions)
    
    # Load dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check required columns
    required_columns = ['url']  # Remove title
    content_columns = ['Article_Content', 'content', 'text']  # Try different possible column names
    
    content_column = None
    for col in content_columns:
        if col in df.columns:
            content_column = col
            break
    
    if content_column is None:
        print(f"Error: No content column found. Available columns: {df.columns.tolist()}")
        return
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Ensure all model-persona columns exist
    df = ensure_columns_exist(df, all_model_versions)
    
    print(f"\nProcessing {len(df)} articles...")
    print(f"Using content column: {content_column}")

    for i, row in df.iterrows():
        url = row['url']
        text = row[content_column]
        
        print(f"\nProcessing article {i+1}/{len(df)}")
        print(f"URL: {url}")
        print(f"Text preview: {str(text)[:100]}..." if len(str(text)) > 100 else f"Text: {text}")
        
        row_updated = False

        # Process ChatGPT models
        if chatgpt_model_version_list:
            responses_needed = False
            for model_version in chatgpt_model_version_list:
                for persona in personas:
                    model_persona_key = f"{model_version}_{persona}"
                    
                    # Check if annotation already exists
                    if pd.isna(df.at[i, model_persona_key]) or df.at[i, model_persona_key] == "":
                        responses_needed = True
                        break
                if responses_needed:
                    break

            if responses_needed:
                if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
                    print("Text is too short or invalid, using empty result")
                    empty_result = create_empty_result_json()
                    for model_version in chatgpt_model_version_list:
                        for persona in personas:
                            model_persona_key = f"{model_version}_{persona}"
                            if pd.isna(df.at[i, model_persona_key]) or df.at[i, model_persona_key] == "":
                                df.at[i, model_persona_key] = empty_result
                                row_updated = True
                else:
                    try:
                        responses = create_chatgpt_content(query, text, chatgpt_model_version_list)
                        for model_version in chatgpt_model_version_list:
                            for persona in personas:
                                model_persona_key = f"{model_version}_{persona}"
                                if pd.isna(df.at[i, model_persona_key]) or df.at[i, model_persona_key] == "":
                                    response = responses.get(model_persona_key, create_empty_result_json())
                                    df.at[i, model_persona_key] = response
                                    row_updated = True
                    except Exception as e:
                        print(f"Error processing ChatGPT models: {e}")

        # Process Claude models
        if claude_model_version_list:
            responses_needed = False
            for model_version in claude_model_version_list:
                for persona in personas:
                    model_persona_key = f"{model_version}_{persona}"
                    
                    # Check if annotation already exists
                    if pd.isna(df.at[i, model_persona_key]) or df.at[i, model_persona_key] == "":
                        responses_needed = True
                        break
                if responses_needed:
                    break

            if responses_needed:
                if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
                    print("Text is too short or invalid, using empty result")
                    empty_result = create_empty_result_json()
                    for model_version in claude_model_version_list:
                        for persona in personas:
                            model_persona_key = f"{model_version}_{persona}"
                            if pd.isna(df.at[i, model_persona_key]) or df.at[i, model_persona_key] == "":
                                df.at[i, model_persona_key] = empty_result
                                row_updated = True
                else:
                    try:
                        responses = create_claude_content(query, text, claude_model_version_list)
                        for model_version in claude_model_version_list:
                            for persona in personas:
                                model_persona_key = f"{model_version}_{persona}"
                                if pd.isna(df.at[i, model_persona_key]) or df.at[i, model_persona_key] == "":
                                    response = responses.get(model_persona_key, create_empty_result_json())
                                    df.at[i, model_persona_key] = response
                                    row_updated = True
                    except Exception as e:
                        print(f"Error processing Claude models: {e}")

        # Save progress after each row
        if row_updated:
            try:
                df.to_csv(annotation_file_path, index=False)
                print(f"Updated results saved to {annotation_file_path}")
            except Exception as e:
                print(f"Error saving results: {e}")

    print(f"\nFinished processing {annotation_file_path}")
    print(f"{'-'*80}")
    return df


def create_annotation_path(original_file_path, annotation_base_dir="annotation_results"):
    """
    Create annotation file path by replicating the directory structure 
    under a different base directory
    """
    # Get relative path from current directory
    rel_path = os.path.relpath(original_file_path, current_dir)
    
    # Create new path under annotation_results directory
    annotation_file_path = os.path.join(current_dir, annotation_base_dir, rel_path)
    
    # Add 'annotated_' prefix to filename
    dir_name = os.path.dirname(annotation_file_path)
    file_name = os.path.basename(annotation_file_path)
    name, ext = os.path.splitext(file_name)
    annotated_file_name = f"annotated_{name}{ext}"
    
    return os.path.join(dir_name, annotated_file_name)


if __name__ == '__main__':
    # Model configurations (2 models)
    claude_model_version_list = [
        'claude-sonnet-4-20250514'
    ]
    chatgpt_model_version_list = [
        'gpt-4.1',
    ]
    other_model_version_list = []  # Not used in this setup

    # Find CSV files in datasets folder
    datasets_path = os.path.join(current_dir, '../c4_datasets')
    # csv_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]
    
    # # Select the most recent file (since filenames contain timestamps)
    # csv_files.sort(reverse=True)
    input_filename = input_filename
    print(f"Selected CSV file: {input_filename}")
    
    file_path = os.path.join(datasets_path, input_filename)
    
    # Set query topic to "gun" (since datasets CSV is gun-related)
    # Drug legalization controversy
    query = query
    
    # Create annotation_datasets directory
    annotation_datasets_path = os.path.join(current_dir, '../annotation_datasets')
    os.makedirs(annotation_datasets_path, exist_ok=True)
    
    # Remove extension from filename
    base_filename = os.path.splitext(input_filename)[0]
    
    # Calculate total annotations per row
    total_models = len(claude_model_version_list) + len(chatgpt_model_version_list)
    total_personas = len(personas)
    annotations_per_row = total_models * total_personas
    
    print(f"=== Annotation Configuration ===")
    print(f"Input file: {file_path}")
    print(f"Query topic: {query}")
    print(f"Claude models ({len(claude_model_version_list)}): {claude_model_version_list}")
    print(f"ChatGPT models ({len(chatgpt_model_version_list)}): {chatgpt_model_version_list}")
    print(f"Personas ({len(personas)}): {personas}")
    print(f"Total annotations per row: {total_models} models × {total_personas} personas = {annotations_per_row}")
    print(f"{'='*50}")
    
    # Verify input file exists
    if not os.path.exists(file_path):
        print(f"Error: Input file not found: {file_path}")
        exit(1)
    
    # Process single annotation
    print(f"\n🔄 Starting annotation process")
    print(f"{'='*60}")
    
    # Generate output filename (without repetition number)
    annotation_filename = f"annotated_{base_filename}.csv"
    annotation_file_path = os.path.join(annotation_datasets_path, annotation_filename)
    
    print(f"Output file: {annotation_file_path}")
    
    # Run annotation process
    result_df = get_df(claude_model_version_list, chatgpt_model_version_list, 
                        other_model_version_list, file_path, annotation_file_path)
    
    if result_df is not None:
        print(f"\n✅ Annotation completed successfully!")
        print(f"Final dataset shape: {result_df.shape}")
        print(f"Results saved to: {annotation_file_path}")
    else:
        print(f"\n❌ Annotation failed!")
    
    print(f"\n🎉 Annotation process completed!")
    print(f"Result file: {annotation_file_path}")