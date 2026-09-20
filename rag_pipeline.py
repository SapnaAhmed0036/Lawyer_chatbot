import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

load_dotenv()

FAISS_DB_PATH = "vector_store/db_faiss_new"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"

# Load FAISS database with mxbai-embed-large (Ollama)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

faiss_db = FAISS.load_local(
    FAISS_DB_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

print(f"✓ Vector database loaded: {faiss_db.index.ntotal} documents")
print(f"✓ Using PURE SEMANTIC SEARCH with {OLLAMA_EMBED_MODEL}")

# Configure LLM for answer generation
# Option 1: Gemini (FASTEST but limited to 20/day)
# Option 2: Groq (FAST & 14,400 requests/day)
# Option 3: Ollama (SLOWER but UNLIMITED)

USE_GEMINI = False  # Quota exhausted
USE_GROQ = True     # Using new Groq API with openai/gpt-oss-120b

if USE_GEMINI:
    import google.generativeai as genai
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("WARNING: GEMINI_API_KEY not found!")
        exit(1)
    
    genai.configure(api_key=gemini_api_key)
    from langchain_google_genai import ChatGoogleGenerativeAI
    answer_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,  # Increased from 0.3 for more natural responses
        google_api_key=gemini_api_key
    )
    print("OK - Gemini Flash 1.5 (FASTEST - 1500 req/day, 1M tokens/min)")

elif USE_GROQ:
    from langchain_groq import ChatGroq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY not found!")
        exit(1)
    
    # Use latest available Groq model
    answer_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        groq_api_key=groq_api_key
    )
    print("✓ Groq openai/gpt-oss-120b (120B parameters, HIGH ACCURACY)")
else:
    from langchain_ollama import OllamaLLM
    answer_llm = OllamaLLM(model="llama3.2:3b", temperature=0.3)
    print("OK - Ollama llama3.2:3b (SLOWER but UNLIMITED)")

PROMPT_TEMPLATE = """
You are a friendly and knowledgeable legal assistant helping people understand Pakistani Criminal Law (PPC 1860 & CrPC 1898). Explain things in a natural, conversational way.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

HOW TO ANSWER:
1. **Talk naturally** - Write like you're explaining to a friend, not reading from a textbook
2. **Be helpful** - Give the answer they need, not just legal jargon
3. **Keep it simple** - Use everyday language, explain legal terms when needed
4. **Be brief** - 2-3 sentences is perfect, don't overwhelm them
5. **Mention ONLY relevant sections** - Only include section numbers that DIRECTLY answer the question (1-2 sections maximum)

CRITICAL: Only mention sections that are ESSENTIAL to answer the question. Don't list every section from the context.

GOOD EXAMPLES (Natural & Focused):

Question: What is extortion?
Answer: Extortion is when someone threatens you to make you hand over money or property. According to Section 383 PPC, it's putting someone in fear of injury to force them to give up something valuable. If someone does this, they can face up to 3 years in prison or a fine under Section 384 PPC.

Question: What is the role of police in criminal investigation?
Answer: When a serious crime, known as a cognizable offense, happens, the police are the ones who step in to investigate. They go to the crime scene, gather evidence, question witnesses, and arrest suspects to figure out what happened. They must complete their investigation without unnecessary delay and then send a detailed report to the Magistrate, as outlined in Section 156 of the CrPC.

Question: What is a Magistrate?
Answer: A Magistrate is basically a judge who handles criminal cases. They have the authority to hear cases, make decisions about bail and arrests, and pass judgments. Think of them as the judicial officers who run the criminal courts under the CrPC.

BAD EXAMPLES (Too many sections - Don't do this):

❌ "...as outlined in Sections 156, 160, 161, 165, and 171 of the CrPC" (TOO MANY!)
❌ "Based on Section 180, 187, 194, 205, 162 CrPC..." (TOO MANY!)
✅ "...as outlined in Section 156 of the CrPC" (PERFECT!)

Now answer the question in a natural, helpful way with MINIMAL section references:
"""

def semantic_retrieval(query, top_k=3):
    """Pure semantic search with smart query enhancement for legal documents."""
    query_lower = query.lower()
    
    # Detect specific section numbers in query
    section_match = re.search(r'\b(section\s+)?(\d+[A-Z]?)\b', query_lower, re.IGNORECASE)
    requested_section = section_match.group(2) if section_match else None
    
    # Detect if question is about procedure (CrPC) or crime/punishment (PPC)
    procedure_keywords = [
        "fir", "police", "arrest", "bail", "procedure", "investigation",
        "warrant", "magistrate", "court", "detain", "cognizable", "trial",
        "information", "cognisable", "how to", "process", "file", "register",
        "role", "powers", "duties"
    ]
    is_procedure = any(kw in query_lower for kw in procedure_keywords)
    
    # Crime-specific keywords
    crime_keywords = {
        "theft": ["theft", "steal", "stolen", "chori", "379"],
        "murder": ["murder", "qatl", "kill", "killing", "death", "302"],
        "hurt": ["hurt", "injury", "wound", "injure", "323", "325"],
        "robbery": ["robbery", "dacoity", "loot", "392"],
        "kidnapping": ["kidnap", "abduct", "363", "364"],
        "assault": ["assault", "attack", "force", "351", "352"],
        "extortion": ["extortion", "threat", "383", "384"],
        "cheating": ["cheat", "fraud", "deceive", "415", "417"],
        "mischief": ["mischief", "damage", "destroy", "425", "426"],
        "trespass": ["trespass", "enter", "441", "447"],
    }
    
    detected_crime = None
    for crime, keywords in crime_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected_crime = crime
            break
    
    # Build enhanced query for better semantic matching
    query_parts = [query]
    
    if requested_section:
        query_parts.append(f"Section {requested_section}")
    
    if is_procedure:
        query_parts.append("Code of Criminal Procedure CrPC procedure process")
    elif detected_crime:
        query_parts.append("Pakistan Penal Code PPC punishment offense crime")
    
    enhanced_query = " ".join(query_parts)
    
    # Semantic search with FEWER results initially (reduced from top_k * 4 to top_k * 2)
    results = faiss_db.similarity_search_with_score(enhanced_query, k=top_k * 2)
    
    # Re-rank results based on relevance
    scored_docs = []
    for doc, distance in results:
        score = 1.0 / (1.0 + distance)  # Convert distance to similarity score
        
        source = doc.metadata.get("source_name", "")
        section = doc.metadata.get("section_number", "")
        provision = doc.metadata.get("provision_type", "GENERAL")
        
        # BOOST 1: Exact section number match (HIGHEST PRIORITY)
        if requested_section and requested_section in section:
            score *= 20.0  # Increased from 10.0
        
        # BOOST 2: Source type match (CrPC for procedure, PPC for crimes)
        if is_procedure and "CrPC" in source:
            score *= 3.0  # Increased from 2.0
        elif not is_procedure and "PPC" in source:
            score *= 3.0  # Increased from 2.0
        
        # BOOST 3: Crime-specific section numbers
        crime_sections = {
            "theft": ["379", "380", "381"],
            "murder": ["302", "300", "299"],
            "hurt": ["323", "324", "325", "326"],
            "robbery": ["392", "393", "394"],
            "kidnapping": ["363", "364", "365"],
            "assault": ["351", "352", "353"],
            "extortion": ["383", "384", "385"],
            "cheating": ["415", "417", "418"],
            "mischief": ["425", "426", "427"],
            "trespass": ["441", "447", "448"],
        }
        
        if detected_crime and detected_crime in crime_sections:
            for crime_sec in crime_sections[detected_crime]:
                if crime_sec in section:
                    score *= 5.0  # Increased from 3.0
                    break
        
        # BOOST 4: Prefer BASE provisions over AGGRAVATED for general questions
        if provision == "BASE" and not any(word in query_lower for word in ["aggravated", "enhanced", "dwelling", "weapon"]):
            score *= 2.0  # Increased from 1.5
        
        scored_docs.append((doc, score))
    
    # Sort by score (highest first) and return top_k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

def format_docs(docs):
    """Format documents with provision type sorting"""
    formatted_chunks = []
    for d in docs:
        source_name = d.metadata.get('source_name', 'Unknown')
        section_number = d.metadata.get('section_number', 'General Provision')
        provision_type = d.metadata.get('provision_type', 'GENERAL')
        
        formatted_chunk = f"[{provision_type} PROVISION - {source_name} - {section_number}]\n{d.page_content}"
        formatted_chunks.append(formatted_chunk)
    
    return "\n\n".join(formatted_chunks)

def build_citations(docs):
    """Build concise citations - only include UNIQUE sections"""
    citations = []
    seen_sections = set()
    
    for doc in docs:
        section = doc.metadata.get("section_number", "")
        if not section or section == "General Provision":
            continue
        
        # Skip if already seen
        if section in seen_sections:
            continue
        seen_sections.add(section)
        
        source_name = doc.metadata.get("source_name", "")
        if "Penal Code" in source_name or "PPC" in source_name:
            book = "PPC"
        elif "Criminal Procedure" in source_name or "CrPC" in source_name:
            book = "CrPC"
        else:
            book = ""

        citation = f"{section}, {book}".strip().strip(",")
        if citation not in citations:
            citations.append(citation)

    if not citations:
        return ""
    
    # Limit to top 3 most relevant sections
    return "Citation: " + ", ".join(citations[:3])

# Create retriever for compatibility
class SemanticRetriever:
    def invoke(self, query):
        return semantic_retrieval(query, top_k=3)

retriever = SemanticRetriever()

def get_answer_text(query):
    """Get answer using Gemini with semantic retrieval"""
    
    # STEP 1: Check if query is legal-related
    query_lower = query.lower()
    irrelevant_keywords = [
        "hello", "hi", "hey", "how are you", "what's up", "good morning", 
        "good evening", "thanks", "thank you", "bye", "goodbye",
        "who are you", "what is your name", "weather", "joke", "story",
        "sing", "poem", "recipe", "movie", "game", "sports", "cricket"
    ]
    
    legal_keywords = [
        "section", "ppc", "crpc", "law", "crime", "offense", "offence",
        "punishment", "bail", "fir", "police", "court", "magistrate",
        "murder", "theft", "robbery", "kidnapping", "assault", "hurt",
        "arrest", "investigation", "trial", "evidence", "witness",
        "penal", "procedure", "criminal", "judge", "lawyer", "accused"
    ]
    
    # Check if query is clearly irrelevant
    has_legal_term = any(keyword in query_lower for keyword in legal_keywords)
    has_irrelevant_term = any(keyword in query_lower for keyword in irrelevant_keywords)
    
    if has_irrelevant_term and not has_legal_term and len(query.split()) < 15:
        return ("I'm here to help you with questions about Pakistani Criminal Law "
                "(Pakistan Penal Code and Code of Criminal Procedure). "
                "Please feel free to ask me anything about criminal law, procedures, sections, or offenses!")
    
    # STEP 2: Get relevant documents using semantic search
    docs = semantic_retrieval(query, top_k=3)

    if not docs:
        return "I don't have enough information to answer this question based on the available legal documents."
    
    # Format context
    context = format_docs(docs)
    
    # Create prompt
    prompt_text = PROMPT_TEMPLATE.format(context=context, question=query)
    
    # Get answer from LLM
    response = answer_llm.invoke(prompt_text)
    
    # Extract text content from response
    if hasattr(response, 'content'):
        answer = response.content
    else:
        answer = str(response)
    
    # STEP 4: Check if answer says "not available" or "don't have information"
    # If so, DON'T add citations (likely hallucination)
    no_info_phrases = [
        "not available", "don't have", "do not have", "isn't available",
        "not present", "can't find", "cannot find", "no information",
        "unable to", "can only explain", "only goes up to"
    ]
    
    has_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
    
    if has_no_info:
        # Don't add any citations for non-existent sections
        return answer
    
    # STEP 5: Extract sections mentioned in the answer itself
    # Only cite sections that are actually mentioned in the answer
    mentioned_sections = set()
    
    # Find all section numbers mentioned in the answer (e.g., "Section 156", "Sections 156 and 171")
    import re
    section_patterns = [
        r'Section\s+(\d+[A-Z]?)',
        r'Sections\s+(\d+[A-Z]?)\s+(?:and|,)\s+(\d+[A-Z]?)',
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                mentioned_sections.update(match)
            else:
                mentioned_sections.add(match)
    
    # Build citation only from mentioned sections
    if mentioned_sections:
        # Determine source (PPC or CrPC) from retrieved docs
        source_map = {}
        for doc in docs:
            section = doc.metadata.get("section_number", "")
            source_name = doc.metadata.get("source_name", "")
            
            # Extract just the number from "Section 156"
            section_num = re.search(r'(\d+[A-Z]?)', section)
            if section_num:
                section_num = section_num.group(1)
                if "Penal Code" in source_name or "PPC" in source_name:
                    source_map[section_num] = "PPC"
                elif "Criminal Procedure" in source_name or "CrPC" in source_name:
                    source_map[section_num] = "CrPC"
        
        # IMPROVED: Use heuristics if section not found in retrieved docs
        def guess_source(sec_num):
            try:
                # Extract base number (e.g., "292A" -> 292)
                num_match = re.search(r'(\d+)', sec_num)
                if not num_match:
                    return "PPC"
                num = int(num_match.group(1))
                
                # MOST COMMON PPC SECTIONS (Explicitly defined)
                common_ppc = {
                    # Murder & related (299-310)
                    299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
                    311, 312, 313, 314, 315, 316, 317, 318,
                    # Hurt
                    319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 337, 338,
                    # Wrongful restraint/confinement
                    339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
                    # Assault
                    349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
                    # Kidnapping
                    359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
                    # Theft
                    378, 379, 380, 381, 382,
                    # Extortion
                    383, 384, 385, 386, 387, 388, 389,
                    # Robbery/Dacoity
                    390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
                    # Mischief
                    425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438,
                    # Trespass
                    441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460,
                    # Cheating
                    415, 416, 417, 418, 419, 420,
                    # Breach of trust
                    405, 406, 407, 408, 409,
                    # Receiving stolen property
                    410, 411, 412, 413, 414,
                    # Criminal intimidation
                    503, 504, 505, 506, 507, 508, 509, 510,
                    # Obscenity
                    292, 293, 294,
                    # General provisions
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106
                }
                
                # MOST COMMON CrPC SECTIONS (Procedures)
                common_crpc = {
                    # FIR & Investigation
                    22, 154, 155, 156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 170, 173,
                    # Arrest
                    41, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61,
                    # Bail
                    496, 497, 498, 499, 500, 501,
                    # Trial procedures & Cognizance
                    190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
                    238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
                    # Magistrate powers
                    30, 31, 32, 33, 34, 35, 36, 37,
                    # Evidence
                    164, 342, 343, 344, 345,
                    # Appeals
                    417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430,
                    # Misc procedures
                    512, 514, 526, 528, 530, 544, 545, 546, 547, 548, 549, 550, 551, 552,
                    553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565
                }
                
                # PRIORITY 1: Check CrPC first (more specific procedures)
                # CrPC sections are typically procedural keywords
                if num in common_crpc:
                    return "CrPC"
                
                # PRIORITY 2: Check PPC (crimes/punishments)
                if num in common_ppc:
                    return "PPC"
                
                # FALLBACK: Use context clues
                # If num > 511, likely CrPC (PPC ends at ~511)
                if num > 511:
                    return "CrPC"
                
                # Otherwise default to PPC (most crimes are in PPC)
                return "PPC"
            except:
                return "PPC"
        
        # Build citation
        citations = []
        for sec_num in sorted(mentioned_sections, key=lambda x: int(re.search(r'\d+', x).group())):
            # First try to get from retrieved docs (MOST RELIABLE)
            source = source_map.get(sec_num)
            
            # If not found, use heuristics
            if not source:
                source = guess_source(sec_num)
            
            citations.append(f"Section {sec_num}, {source}")
        
        citation_text = "Citation: " + ", ".join(citations)
    else:
        # Fallback to old method if no sections mentioned
        citation_text = build_citations(docs)
    
    # Add citation if not already present
    if citation_text and "citation:" not in answer.lower():
        answer = answer.rstrip() + "\n\n" + citation_text

    return answer

