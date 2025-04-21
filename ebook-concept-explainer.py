"""
EbookExplainer - A tool for extracting and explaining concepts from ebook chapters

This module provides functionality to:
1. Extract text from various ebook formats (PDF, EPUB)
2. Identify key concepts in a specified chapter
3. Generate simple, accessible explanations with real-life examples
4. Output the explanations in a readable format
"""

import os
import argparse
import logging
import json
from pathlib import Path
import ollama
import roman
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EbookExplainer")

class EbookExplainer:
    def __init__(self, ebook_path, output_dir=None, model_name="llama3", log_level="INFO"):
        """Initialize the EbookExplainer with configuration parameters."""
        self.ebook_path = os.path.abspath(ebook_path)
        self.ebook_format = self._determine_format(ebook_path)
        
        # Create output directory based on ebook name by default
        if output_dir is None:
            ebook_name = os.path.splitext(os.path.basename(ebook_path))[0]
            self.output_dir = os.path.join(os.path.dirname(ebook_path), f"{ebook_name}_explanations")
        else:
            self.output_dir = output_dir
            
        self.model_name = model_name
        
        # Set up output directory
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        
        # Configure logging
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "explainer.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(getattr(logging, log_level))
        
        logger.info(f"Ebook: {self.ebook_path} (Format: {self.ebook_format})")
        logger.info(f"Explanations will be saved to: {self.output_dir}")
        
        # Load ebook content
        self.book_content = self._load_ebook()
        self.chapters = self._identify_chapters()

    def _determine_format(self, path):
        """Determine the ebook format based on file extension."""
        extension = os.path.splitext(path)[1].lower()
        if extension == '.pdf':
            return 'pdf'
        elif extension == '.epub':
            return 'epub'
        else:
            raise ValueError(f"Unsupported ebook format: {extension}")

    def _load_ebook(self):
        """Load the ebook content based on its format."""
        logger.info(f"Loading ebook: {self.ebook_path}")
        try:
            if self.ebook_format == 'pdf':
                return self._load_pdf()
            elif self.ebook_format == 'epub':
                return self._load_epub()
        except Exception as e:
            logger.error(f"Error loading ebook: {e}")
            raise

    def _load_pdf(self):
        """Load content from a PDF file."""
        try:
            document = fitz.open(self.ebook_path)
            content = []
            
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                content.append({
                    'page_num': page_num + 1,
                    'text': page.get_text()
                })
                
            logger.info(f"Loaded PDF with {len(content)} pages")
            return content
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def _load_epub(self):
        """Load content from an EPUB file."""
        try:
            book = epub.read_epub(self.ebook_path)
            content = []
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    content.append({
                        'id': item.id,
                        'text': text
                    })
                    
            logger.info(f"Loaded EPUB with {len(content)} sections")
            return content
        except Exception as e:
            logger.error(f"Error loading EPUB: {e}")
            raise

    def _identify_chapters(self):
        """Identify chapter boundaries in the ebook."""
        chapters = []
        
        if self.ebook_format == 'pdf':
            # For PDF: Use a heuristic approach to identify chapter headings
            chapter_pattern = re.compile(r'(?:Chapter|CHAPTER)\s+(\d+|[IVXLCDM]+)', re.IGNORECASE)
            current_chapter = {"number": 0, "title": "Front Matter", "start": 0, "end": 0, "content": ""}
            
            for page in self.book_content:
                page_text = page['text']
                chapter_matches = chapter_pattern.findall(page_text)
                
                if chapter_matches:
                    # Complete the previous chapter
                    if current_chapter["number"] != 0:  # Check if not front matter
                        try:
                            chapter_num = int(current_chapter["number"]) if isinstance(current_chapter["number"], str) and current_chapter["number"].isdigit() else current_chapter["number"]
                            if isinstance(chapter_num, int) and chapter_num > 0:
                                current_chapter["end"] = page['page_num'] - 1
                                chapters.append(current_chapter)
                        except (ValueError, TypeError):
                            logger.warning(f"Skipping invalid previous chapter number: {current_chapter['number']}")
                    
                    # Start a new chapter
                    chapter_num = chapter_matches[0]
                    logger.debug(f"Found chapter match: {chapter_num}")
                    
                    # Validate and convert chapter number
                    try:
                        if chapter_num.isdigit():
                            chapter_num = int(chapter_num)  # Numeric chapter
                        else:
                            # Try Roman numeral conversion
                            chapter_num = roman.fromRoman(chapter_num.upper())
                        # Extract chapter title
                        title_match = re.search(f"(?:Chapter|CHAPTER)\s+{chapter_num}[:\.\s]+(.+?)(?:\n|\r|$)", page_text, re.IGNORECASE)
                        title = title_match.group(1).strip() if title_match else f"Chapter {chapter_num}"
                        
                        current_chapter = {
                            "number": chapter_num,
                            "title": title,
                            "start": page['page_num'],
                            "end": None,
                            "content": page_text
                        }
                    except (ValueError, roman.InvalidRomanNumeralError):
                        logger.warning(f"Skipping invalid chapter number: {chapter_num}")
                        continue  # Skip invalid chapter numbers
                    
                else:
                    # Append content to current chapter
                    if current_chapter["number"] != 0:
                        current_chapter["content"] += "\n" + page_text
            
            # Handle the last chapter
            if current_chapter["number"] != 0:
                try:
                    chapter_num = int(current_chapter["number"]) if isinstance(current_chapter["number"], str) and current_chapter["number"].isdigit() else current_chapter["number"]
                    if isinstance(chapter_num, int) and chapter_num > 0:
                        current_chapter["end"] = self.book_content[-1]['page_num']
                        chapters.append(current_chapter)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid last chapter number: {current_chapter['number']}")
                    
        elif self.ebook_format == 'epub':
            # For EPUB: Look for chapter identifiers in the content
            chapter_pattern = re.compile(r'(?:Chapter|CHAPTER)\s+(\d+|[IVXLCDM]+)', re.IGNORECASE)
            
            for i, section in enumerate(self.book_content):
                text = section['text']
                chapter_matches = chapter_pattern.findall(text)
                
                if chapter_matches:
                    for match in chapter_matches:
                        # Validate and convert chapter number
                        try:
                            if match.isdigit():
                                chapter_num = int(match)
                            else:
                                chapter_num = roman.fromRoman(match.upper())  # Roman numeral
                            
                            # Extract chapter title
                            title_match = re.search(f"(?:Chapter|CHAPTER)\s+{match}[:\.\s]+(.+?)(?:\n|\r|$)", text, re.IGNORECASE)
                            title = title_match.group(1).strip() if title_match else f"Chapter {match}"
                            
                            chapters.append({
                                "number": chapter_num,
                                "title": title,
                                "section_id": section['id'],
                                "content": text
                            })
                        except (ValueError, roman.InvalidRomanNumeralError):
                            logger.warning(f"Skipping invalid chapter number in EPUB: {match}")
                            continue
        
        logger.info(f"Identified {len(chapters)} chapters")
        return chapters

    def get_chapter_content(self, chapter_num):
        """Get the content of a specific chapter by number."""
        try:
            # Try to convert input to integer or Roman numeral
            if chapter_num.isdigit():
                target_num = int(chapter_num)
            else:
                target_num = roman.fromRoman(chapter_num.upper())
            
            for chapter in self.chapters:
                chapter_num = chapter["number"]
                if isinstance(chapter_num, int) and chapter_num == target_num:
                    return chapter
                elif isinstance(chapter_num, str):
                    try:
                        if roman.fromRoman(chapter_num.upper()) == target_num:
                            return chapter
                    except roman.InvalidRomanNumeralError:
                        continue
            
            # If chapter not found by exact number, try finding closest match
            available_nums = []
            for ch in self.chapters:
                if isinstance(ch["number"], int):
                    available_nums.append(ch["number"])
                elif isinstance(ch["number"], str):
                    try:
                        available_nums.append(roman.fromRoman(ch["number"].upper()))
                    except roman.InvalidRomanNumeralError:
                        continue
            
            if available_nums:
                closest = min(available_nums, key=lambda x: abs(x - target_num))
                logger.warning(f"Chapter {chapter_num} not found. Using closest match: Chapter {closest}")
                return next(ch for ch in self.chapters if ch["number"] == closest or (isinstance(ch["number"], str) and roman.fromRoman(ch["number"].upper()) == closest))
        
        except (ValueError, roman.InvalidRomanNumeralError):
            logger.error(f"Invalid chapter number: {chapter_num}")
        
        logger.error(f"Chapter {chapter_num} not found")
        return None

    def extract_concepts(self, chapter_content):
        """Extract key concepts from chapter content using LLM."""
        logger.info(f"Extracting concepts from chapter: {chapter_content['title']}")
        
        # Create a prompt for concept extraction
        extract_prompt = f"""
You are an expert knowledge extractor. From the following book chapter text, identify the 5-10 most important concepts.
Focus on terminology, theories, principles, and key ideas that would be essential for understanding this chapter.
For each concept, provide:
1. The concept name
2. A brief identification of its importance (why is this a key concept?)

CHAPTER TITLE: {chapter_content['title']}

CHAPTER CONTENT (excerpt):
{chapter_content['content'][:10000]}  # Use first 10000 chars to fit context window

OUTPUT FORMAT:
List of concepts in JSON format:
[
  {{"concept": "concept name", "importance": "brief statement of importance"}}
]
"""
        
        try:
            response = ollama.generate(model=self.model_name, prompt=extract_prompt)
            concepts_text = response["response"]
            
            # Extract JSON from response (handling case where LLM includes explanatory text)
            json_match = re.search(r'\[\s*\{.*\}\s*\]', concepts_text, re.DOTALL)
            if json_match:
                concepts_json = json_match.group(0)
                concepts = json.loads(concepts_json)
            else:
                # Fallback if JSON extraction fails
                logger.warning("Could not extract JSON from response. Processing raw output.")
                concepts = self._parse_concepts_from_text(concepts_text)
                
            logger.info(f"Extracted {len(concepts)} concepts")
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []
            
    def _parse_concepts_from_text(self, text):
        """Fallback parser for when LLM doesn't generate proper JSON."""
        concepts = []
        lines = text.strip().split('\n')
        current_concept = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for numbered or bullet points for concepts
            concept_match = re.match(r'(?:\d+\.|\*|\-)\s*(?:"([^"]+)"|([^:]+)):', line)
            if concept_match:
                if current_concept:
                    concepts.append(current_concept)
                    
                concept_name = concept_match.group(1) or concept_match.group(2)
                importance = line.split(':', 1)[1].strip() if ':' in line else ""
                current_concept = {"concept": concept_name.strip(), "importance": importance}
            elif current_concept:
                # Append to current concept's importance
                current_concept["importance"] += " " + line
        
        # Add the last concept
        if current_concept:
            concepts.append(current_concept)
            
        return concepts

    def explain_concepts(self, concepts, chapter_title):
        """Generate beginner-friendly explanations for extracted concepts."""
        logger.info(f"Generating explanations for {len(concepts)} concepts")
        
        explanations = []
        for concept in concepts:
            logger.info(f"Explaining concept: {concept['concept']}")
            
            explain_prompt = f"""
You are an exceptional teacher who can explain complex topics to complete beginners.
Explain the following concept from the chapter "{chapter_title}" in simple, accessible language.

CONCEPT: {concept['concept']}
CONTEXT: {concept['importance']}

Your explanation should:
1. Define the concept in everyday language
2. Use 1-2 concrete, real-life examples or analogies that make it easy to understand
3. Explain why this concept matters in practical terms
4. Avoid jargon without explaining it first
5. Keep your explanation under 300 words

Write this as if you're explaining it to a smart 15-year-old who has no background in this subject.
"""
            
            try:
                response = ollama.generate(model=self.model_name, prompt=explain_prompt)
                explanation = response["response"].strip()
                
                explanations.append({
                    "concept": concept['concept'],
                    "importance": concept['importance'],
                    "explanation": explanation
                })
                
            except Exception as e:
                logger.error(f"Error explaining concept '{concept['concept']}': {e}")
                explanations.append({
                    "concept": concept['concept'],
                    "importance": concept['importance'],
                    "explanation": f"Error generating explanation: {e}"
                })
        
        return explanations

    def format_output(self, explanations, chapter_title, chapter_num):
        """Format the explanations into markdown and JSON outputs."""
        # Create markdown output
        markdown_file = os.path.join(self.output_dir, f"chapter_{chapter_num}_explanations.md")
        json_file = os.path.join(self.output_dir, f"chapter_{chapter_num}_explanations.json")
        
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(f"# Explanations for Chapter {chapter_num}: {chapter_title}\n\n")
            f.write("*This document explains key concepts from the chapter in beginner-friendly terms with real-life examples.*\n\n")
            
            for i, item in enumerate(explanations, 1):
                f.write(f"## {i}. {item['concept']}\n\n")
                f.write(f"{item['explanation']}\n\n")
                f.write("---\n\n")
        
        # Create JSON output
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "explanations": explanations
            }, f, indent=2)
        
        logger.info(f"Saved explanations to {markdown_file} and {json_file}")
        return markdown_file

    def explain_chapter(self, chapter_num):
        """Process a single chapter: extract concepts and generate explanations."""
        # Get chapter content
        chapter = self.get_chapter_content(chapter_num)
        if not chapter:
            return None
        
        logger.info(f"Processing Chapter {chapter_num}: {chapter['title']}")
        
        # Extract key concepts
        concepts = self.extract_concepts(chapter)
        
        # Generate explanations
        explanations = self.explain_concepts(concepts, chapter['title'])
        
        # Format and save output
        output_file = self.format_output(explanations, chapter['title'], chapter_num)
        
        return {
            "chapter": chapter['title'],
            "concepts": concepts,
            "explanations": explanations,
            "output_file": output_file
        }

def main():
    """Parse command line arguments and run the ebook explainer."""
    parser = argparse.ArgumentParser(description="Ebook Concept Explainer Tool")
    parser.add_argument("--ebook", "-e", required=True, help="Path to the ebook file (PDF or EPUB)")
    parser.add_argument("--chapter", "-c", required=True, help="Chapter number to explain")
    parser.add_argument("--output", "-o", default=None, 
                        help="Output directory for explanations (defaults to ebook_name_explanations)")
    parser.add_argument("--model", "-m", default="llama3", help="Ollama model for explanations")
    parser.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    try:
        explainer = EbookExplainer(
            ebook_path=args.ebook,
            output_dir=args.output,
            model_name=args.model,
            log_level=args.log_level
        )
        result = explainer.explain_chapter(args.chapter)
        
        if result:
            print(f"\nExplanation complete!")
            print(f"Chapter: {result['chapter']}")
            print(f"Concepts explained: {len(result['explanations'])}")
            print(f"Output saved to: {result['output_file']}")
        else:
            print(f"\nFailed to explain chapter {args.chapter}. Check the logs for details.")
            
    except KeyboardInterrupt:
        logger.info("Explanation process interrupted by user")
    except Exception as e:
        logger.error(f"Error running explainer: {e}", exc_info=True)

if __name__ == "__main__":
    main()
