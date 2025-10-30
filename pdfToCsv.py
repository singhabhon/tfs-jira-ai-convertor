import time
import base64
import csv
import io
import json
import os
import re
import shutil
import socket
import pandas as pd
import requests
import tempfile
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm
from urllib.parse import quote, urlencode, urlparse, urlunparse
import argparse
import datetime

# -------------------------------------------------
# TIMER START
# -------------------------------------------------
start_time = time.time()

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
pdf_path = "confluence_testcases.pdf"
ai_output_csv = "ai_extracted_testcases.csv"
sample_xray_csv = "sample_xray_format.csv"
output_csv = "xray_output.csv"
pages_per_chunk = 5  # You can adjust this as needed

# -------------------------------------------------
# CONFIGURABLE SUMMARY PREFIX
# -------------------------------------------------
summary_prefix = "TestSummary_ModuleName"

# -------------------------------------------------
# OpenAI Configuration
# -------------------------------------------------
OPENAI_API_KEY = ""  # Add your OpenAI API key here
OPENAI_MODEL = "gpt-5-mini"  # or your preferred model
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# -------------------------------------------------
# Column Mappings
# -------------------------------------------------
xray_columns = [
    "Summary(Test Case Name)",
    "Description",
    "Test Case Identifier",
    "Action",
    "Expected Result"
]

mapping = {
    "Test Scenario": "Summary(Test Case Name)",
    "Test Case Description": "Description",
    "Test Case ID": "Test Case Identifier",
    "Test Steps": "Action",
    "Expected Result": "Expected Result"
}

new_column_order = [
    "Summary(Test Case Name)",
    "Description",
    "Test Case Identifier",
    "Action",
    "Expected Result"
]

# Function to build PDF message
def build_pdf_message(pdf_path, prompt_text):
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    print(f"[PDF] base64 head={b64[:60]}...")
    content = [
        {
            "type": "input_file",
            "mime_type": "application/pdf",
            "data": b64
        },
        {
            "type": "input_text",
            "text": prompt_text + "\n\n(If file cannot be read, explicitly say: FILE_UNREADABLE)"
        }
    ]
    return {"role": "user", "content": content}

# -------------------------------------------------
# LLM CALL
# -------------------------------------------------
def call_llm_text(messages):
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': OPENAI_MODEL,
        'input': messages,
        'max_output_tokens': 32000
    }
    print(f"Sending request with max_output_tokens: {payload['max_output_tokens']}")
    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    response_data = resp.json()
    
    if response_data.get("output_text"):
        return response_data["output_text"]
    
    for item in response_data.get("output", []):
        for piece in item.get("content", []):
            if piece.get("type") == "output_text":
                return piece.get("text", "")
    
    raise ValueError("No text content returned from OpenAI response.")

# -------------------------------------------------
# SANITY CHECK PDF
# -------------------------------------------------
def sanity_check_pdf(path):
    if not os.path.exists(path):
        print(f"[PDF] ERROR: File not found: {path}")
        return ""
    
    size = os.path.getsize(path)
    print(f"[PDF] Path={path} size={size} bytes")
    
    try:
        reader = PdfReader(path)
        pages = len(reader.pages)
        print(f"[PDF] Page count (PyPDF2)={pages}")
        
        sample_text = []
        for i in range(min(2, pages)):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                sample_text.append(f"Page {i+1} preview: {text[:100]}...")
        
        if sample_text:
            print("\n".join(sample_text))
            return sample_text[0]
        else:
            print("[PDF] WARNING: No text extracted from sample pages")
            return ""
            
    except Exception as e:
        print(f"[PDF] ERROR: Failed to read PDF: {e}")
        return ""

# -------------------------------------------------
# EXTRACT MARKDOWN TABLE
# -------------------------------------------------
def extract_markdown_table(text):
    """
    Tries to find the first markdown table.
    Handles tables inside code fences too.
    """
    # Remove code fences if present
    fenced = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
    candidates = fenced if fenced else [text]
    
    for blob in candidates:
        lines = blob.splitlines()
        table_lines = []
        in_table = False
        
        for line in lines:
            if line.strip().startswith("|"):
                if not in_table:
                    in_table = True
                table_lines.append(line)
            elif in_table and not line.strip():
                break
            elif in_table:
                # If we hit a non-empty line that doesn't start with |, table is done
                break
        
        if table_lines:
            return "\n".join(table_lines)
    
    return None

def markdown_table_to_dataframe(text):
    if not text:
        raise ValueError("No table text provided.")
    
    # Split into lines and filter empty ones
    lines = [line for line in text.strip().split("\n") if line.strip()]
    
    if len(lines) < 3:  # Need header + separator + at least one row
        raise ValueError("Table must have at least a header row and one data row.")
    
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
    ]
    
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

# --- Validation function ---
def validate_extraction_completeness(df, expected_min_rows=20):
    actual_rows = len(df)
    print(f"📊 Extracted {actual_rows} test cases")
    
    if actual_rows < expected_min_rows:
        print(f"⚠️  Warning: Only {actual_rows} test cases extracted. Expected at least {expected_min_rows}.")
        print("This might indicate incomplete extraction from the PDF.")
        return False
    return True

# -------------------------------------------------
# PDF CHUNKING
# -------------------------------------------------
def split_pdf_by_pages(pdf_path, pages_per_chunk=5):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    chunks = []
    
    for start in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        for i in range(start, min(start + pages_per_chunk, total_pages)):
            writer.add_page(reader.pages[i])
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp_file.name, "wb") as f:
            writer.write(f)
        chunks.append(temp_file.name)
    
    return chunks

# -------------------------------------------------
# CHUNKED EXTRACTION
# -------------------------------------------------
def call_ai_to_extract_testcases_chunked(pdf_path, ai_output_csv, pages_per_chunk=5):
    pt = (
        "You are an expert in optical character recognition (OCR) and table extraction. "
        "Your task is to extract all tables and their content from the attached PDF file.\n\n"
        
        "🚨 CRITICAL INSTRUCTIONS 🚨\n"
        "1. Extract ALL test cases from EVERY page in this PDF chunk\n"
        "2. Tables must be formatted EXACTLY as shown in the example below\n"
        "3. Include the table header row in your response\n"
        "4. ALWAYS start each row with a | character and end with a | character\n"
        "5. Ensure ALL table rows follow the same column order\n\n"
        
        "📝 OUTPUT FORMAT:\n"
        "- You MUST output a properly formatted markdown table with these EXACT headers and in this EXACT order:\n"
        "  | Test Scenario | Test Case Description | Test Case ID | Test Steps | Expected Result |\n"
        "- Your response MUST start with this markdown table - do not include any explanatory text before it\n"
        "- Every row MUST contain the exact | separator character between each column\n\n"
        
        "📋 COLUMN RULES:\n"
        "- Copy each cell's content **exactly as it appears in the PDF**. Do NOT fill down, infer, or copy values for blank cells\n"
        "- If a cell in the PDF is blank, use an empty string in that cell: | ... | | ... |\n"
        "- If a cell in the PDF contains a list (multiple lines), combine all lines into a single cell, separated by `<br>`\n"
        "- Test Case ID format: Preserve ALL characters including underscores, hyphens, and numbers - don't add spaces\n"
        "- Test Steps: Preserve ALL numbering and formatting as it appears in the PDF\n\n"
        
        "🔍 PAGE HANDLING:\n"
        "- Process EVERY page in this chunk - do not stop after the first page\n"
        "- If content spans across page boundaries, reconnect it properly\n"
        "- Look for content in tables, as well as structured lists that may not have visible borders\n\n"
        
        "⚠️ FINAL CHECKS:\n"
        "- Verify your markdown table syntax is correct with proper | separators for all columns and rows\n"
        "- Make sure ALL content from ALL pages is extracted\n"
        "- Do not add any commentary, analysis, or additional text to your response\n\n"
        
        "Here is a sample table (use this format EXACTLY, including the header row and separator row):\n"
        "| Test Scenario | Test Case Description | Test Case ID | Test Steps | Expected Result |\n"
        "|--------------|----------------------|--------------|-----------|----------------|\n"
        "| FastTrack    | Verify user can send a new order | TC_001 | 1. Login to portal<br>2. Navigate to Orders<br>3. Enter order details<br>4. Submit order | 1. User logged in<br>2. Orders page displayed<br>3. Order details accepted<br>4. Order submitted successfully |\n"
        "| FastTrack    | Verify error for invalid input | TC_002 | 1. Login to portal<br>2. Navigate to Orders<br>3. Enter invalid data<br>4. Submit order | 1. User logged in<br>2. Orders page displayed<br>3. Fields marked in red<br>4. Error message displayed |\n"
    )
    
    prompt = pt
    chunk_paths = split_pdf_by_pages(pdf_path, pages_per_chunk)
    all_dfs = []
    temp_csvs = []
    
    for idx, chunk_path in enumerate(chunk_paths):
        print(f"\n🔹 Processing chunk {idx+1}/{len(chunk_paths)}: {chunk_path}")
        chunk_csv = f"chunk_{idx+1}.csv"
        temp_csvs.append(chunk_csv)
        
        try:
            # Use the same extraction logic for each chunk
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"  🔄 Attempt {attempt + 1} for chunk {idx+1}...")
                    messages = [build_pdf_message(chunk_path, prompt)]
                    response_text = call_llm_text(messages)
                    raw_content = response_text
                    print(f"  📊 AI response length: {len(raw_content)} characters")
                    
                    md_table = extract_markdown_table(raw_content)
                    if not md_table:
                        raise ValueError("No markdown table found in AI response.")
                        
                    df = markdown_table_to_dataframe(md_table)
                    
                    if validate_extraction_completeness(df, expected_min_rows=1):  # Accept at least 1 per chunk
                        df.to_csv(chunk_csv, index=False)
                        all_dfs.append(df)
                        print(f"  ✅ Chunk {idx+1} extracted {len(df)} test cases.")
                        
                        # Show elapsed time after every chunk
                        elapsed = time.time() - start_time
                        hours, rem = divmod(elapsed, 3600)
                        minutes, seconds = divmod(rem, 60)
                        print(f"⏱️ Elapsed time after chunk {idx+1}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)")
                        break
                    else:
                        if attempt < max_retries - 1:
                            print("  🔄 Retrying with more explicit instructions...")
                            prompt += f"\n\n🔄 RETRY INSTRUCTION: Previous attempt only extracted {len(df)} test cases. Please ensure you process ALL pages in this chunk."
                            continue
                        else:
                            print("  ⚠️ Proceeding with potentially incomplete extraction for this chunk.")
                            df.to_csv(chunk_csv, index=False)
                            all_dfs.append(df)
                            break
                        
                except Exception as e:
                    print(f"  ❌ Attempt {attempt + 1} failed for chunk {idx+1}: {e}")
                    if attempt == max_retries - 1:
                        print(f"  ❌ Skipping chunk {idx+1} due to repeated failures.")
            
            # Clean up temp PDF
            os.remove(chunk_path)
        except Exception as e:
            print(f"❌ Failed to process chunk {idx+1}: {e}")
    
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv(ai_output_csv, index=False)
        print(f"\n✅ All chunks combined and written to {ai_output_csv}")
        
        # Clean up temp csvs
        for temp_csv in temp_csvs:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    else:
        print("❌ No data extracted from any chunk.")

# -------------------------------------------------
# RUN EXTRACTION (if needed)
# -------------------------------------------------
extraction_ran = False
if not os.path.exists(ai_output_csv):
    call_ai_to_extract_testcases_chunked(pdf_path, ai_output_csv, pages_per_chunk=pages_per_chunk)
    extraction_ran = True
else:
    print(f"📁 AI-extracted CSV already exists: {ai_output_csv}")
    # Check if we should re-extract
    df_existing = pd.read_csv(ai_output_csv, dtype=str)
    print(f"📊 Existing file contains {len(df_existing)} test cases")
    user_input = input("Do you want to re-extract from PDF? (y/n): ").lower().strip()
    if user_input == 'y':
        print("🔄 Re-extracting from PDF...")
        call_ai_to_extract_testcases_chunked(pdf_path, ai_output_csv, pages_per_chunk=pages_per_chunk)

# -------------------------------------------------
# LOAD AI OUTPUT
# -------------------------------------------------
df_confluence = pd.read_csv(ai_output_csv, dtype=str).fillna("")

# Normalize columns: lower-case and strip spaces for case-insensitive matching
df_confluence.columns = [c.strip().lower() for c in df_confluence.columns]

# Helper: map canonical mapping keys to actual columns in df_confluence
def get_actual_column(df_columns, target):
    target_norm = target.strip().lower()
    for col in df_columns:
        if col == target_norm:
            return col
    return None

# Clean placeholders (case-insensitive)
for col in ["test scenario", "test case description", "test steps"]:  # <-- replaced here
    actual_col = get_actual_column(df_confluence.columns, col)
    if actual_col:
        df_confluence[actual_col] = df_confluence[actual_col].replace("-", "")

confluence_rows = df_confluence.to_dict(orient="records")
print(f"[AI] Loaded {len(confluence_rows)} extracted test case rows")

# Check required columns (case-insensitive)
missing = []
for col in mapping:
    if not get_actual_column(df_confluence.columns, col):
        missing.append(col)
if missing:
    raise ValueError(f"Missing required columns in AI output: {missing}")

# -------------------------------------------------
# TRANSFORM TO XRAY FORMAT
# -------------------------------------------------
xray_rows = []

for row in confluence_rows:
    xray_row = {col: "" for col in xray_columns}
    
    # Use case-insensitive mapping from AI output to xray columns
    for source_col, target_col in mapping.items():
        actual_col = get_actual_column(df_confluence.columns, source_col)
        value = row.get(actual_col, "")
        xray_row[target_col] = str(value) if pd.notnull(value) else ""
    
    # Split into multiple rows if test steps exist
    action = xray_row.get("Action", "")
    steps = [
        s.strip() for s in re.split(
            r'\n|\s*<br\s*/?\s*>\s*|\s*\d+\s*[.)]?\s+',
            action,
            flags=re.IGNORECASE
        ).split('\n')
        if s.strip()
    ] if action else [""]
    
    for step in steps:
        expanded = xray_row.copy()
        expanded["Action"] = step
        xray_rows.append(expanded)

print(f"[Transform] Expanded rows (by steps): {len(xray_rows)}")

# Clean multiline fields
def clean_multiline_fields(row):
    for col in ["Action", "Expected Result"]:
        if col in row and isinstance(row[col], str):
            row[col] = re.sub(r'[ \t]+', ' ', row[col].replace('\r', '')).strip()
    return row

xray_rows = [clean_multiline_fields(r) for r in xray_rows]
df_xray = pd.DataFrame(xray_rows, columns=xray_columns)
df_xray.columns = [c.strip() for c in df_xray.columns]

# Blank out Description & Summary except first occurrence per Test Case Identifier
desc_col = "Description"
summary_col = "Summary(Test Case Name)"
tcid_col = "Test Case Identifier"

if all(c in df_xray.columns for c in [desc_col, summary_col, tcid_col]):
    mask = df_xray.duplicated(subset=[tcid_col])
    df_xray.loc[mask, desc_col] = ""
    df_xray.loc[mask, summary_col] = ""

# Prefix summary
if summary_col in df_xray.columns:
    def prefix_summary(val):
        if pd.notnull(val) and str(val).strip():
            return f"{summary_prefix}{val}"
        return val
    df_xray[summary_col] = df_xray[summary_col].apply(prefix_summary)

# Replace <br> globally
def replace_br(val):
    if isinstance(val, str):
        return re.sub(r'<br\s*/?>', '\n', val)
    return val

df_xray = df_xray.applymap(replace_br)

# Consolidate expected result only in last row
def process_expected_result_group(group):
    er_col = "Expected Result"
    if er_col not in group.columns:
        return group
        
    # Move all expected results to last row
    non_empty_er = group[er_col].dropna().astype(str)
    non_empty_er = [x for x in non_empty_er if x.strip()]
    
    if non_empty_er:
        group[er_col] = ""  # Clear all
        group.iloc[-1, group.columns.get_loc(er_col)] = "\n".join(non_empty_er)
    
    return group

if tcid_col in df_xray.columns:
    df_xray = df_xray.groupby(tcid_col, group_keys=False).apply(process_expected_result_group)

# Column ordering
final_columns = [c for c in xray_columns if c in df_xray.columns] + [
    c for c in new_column_order if c not in xray_columns
]
df_xray = df_xray[final_columns]

# -------------------------------------------------
# Write output
# -------------------------------------------------
df_xray.to_csv(output_csv, index=False)
print(f"✅ Output written to {output_csv}")

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
print(f"\n📋 FINAL PROCESSING SUMMARY:")
print("=" * 50)
print(f"📁 Input PDF: {pdf_path}")
print(f"📊 AI-extracted test cases: {len(confluence_rows)}")
print(f"📈 Final processed rows: {len(xray_rows)}")
print(f"💾 Output file: {output_csv}")
print("=" * 50)

# -------------------------------------------------
# TIMER END
# -------------------------------------------------
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print("\n" + "=" * 50)
print(f"⏱️ Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)")
print("=" * 50)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

raw_pdf_text = extract_text_from_pdf(pdf_path)
print(raw_pdf_text[:1000])  # Preview first 1000 chars

# Fix Test Case Identifier formatting: join split characters and remove <br>, preserve underscores
tcid_col = "Test Case Identifier"
if tcid_col in df_xray.columns:
    def fix_tcid(val):
        if isinstance(val, str):
            # Remove <br> and whitespace, but preserve underscores
            cleaned = re.sub(r'<br\s*/?>', '', val)
            cleaned = re.sub(r'\s+', '', cleaned)
            return cleaned
        return val
    
    df_xray[tcid_col] = df_xray[tcid_col].apply(fix_tcid)

# Write final output
df_xray.to_csv(output_csv, index=False)
print(f"✅ Final output written to {output_csv}")