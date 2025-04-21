import os
import ast
import pygraphviz as pgv
from pathlib import Path
import ollama

# Configuration
REPO_PATH = "C:/Users/zm_if/Desktop/revamp"  # Replace with your repository path
OUTPUT_DOT_FILE = "codebase_flowchart.dot"
OUTPUT_PNG_FILE = "codebase_flowchart.png"
MODEL_NAME = "codellama"  # Ollama model for code understanding

def parse_python_file(file_path):
    """Parse a Python file and extract classes, methods, and functions."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=file_path)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                classes.append({"name": node.name, "methods": methods})
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                functions.append(node.name)
        
        return {
            "file": file_path,
            "classes": classes,
            "functions": functions
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def analyze_codebase(repo_path):
    """Walk through the repository and analyze all Python files."""
    codebase_structure = {}
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                parsed_data = parse_python_file(file_path)
                if parsed_data:
                    codebase_structure[relative_path] = parsed_data
    
    return codebase_structure

def summarize_codebase_with_llm(codebase_structure):
    """Use a local LLM to summarize the codebase and infer what needs to be done."""
    summary_prompt = """
You are an AI assistant analyzing a Python codebase for an investment fund using ML and AI. Below is the structure of the codebase, including modules, classes, methods, and functions. Provide:
1. A high-level summary of the codebase's purpose and functionality.
2. Insights into what has been implemented.
3. Suggestions for what needs to be done next (e.g., missing components, optimizations).

Codebase structure:
"""
    for module, data in codebase_structure.items():
        summary_prompt += f"\nModule: {module}\n"
        if data["classes"]:
            summary_prompt += "Classes:\n"
            for cls in data["classes"]:
                summary_prompt += f"  - {cls['name']} (Methods: {', '.join(cls['methods'])})\n"
        if data["functions"]:
            summary_prompt += f"Functions: {', '.join(data['functions'])}\n"
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=summary_prompt)
        return response["response"]
    except Exception as e:
        return f"Error summarizing with LLM: {e}"

def generate_flowchart(codebase_structure):
    """Generate a flowchart of the codebase using Graphviz."""
    G = pgv.AGraph(directed=True, rankdir="TB", name="Codebase Flowchart")
    
    # Add nodes for each module, class, and method
    for module, data in codebase_structure.items():
        module_node = module.replace("/", "_").replace(".", "_")
        G.add_node(module_node, label=f"Module: {module}", shape="box")
        
        for cls in data["classes"]:
            class_node = f"{module_node}_{cls['name']}"
            G.add_node(class_node, label=f"Class: {cls['name']}", shape="ellipse")
            G.add_edge(module_node, class_node)
            
            for method in cls["methods"]:
                method_node = f"{class_node}_{method}"
                G.add_node(method_node, label=f"Method: {method}", shape="diamond")
                G.add_edge(class_node, method_node)
        
        for func in data["functions"]:
            func_node = f"{module_node}_{func}"
            G.add_node(func_node, label=f"Function: {func}", shape="diamond")
            G.add_edge(module_node, func_node)
    
    # Save the flowchart
    G.write(OUTPUT_DOT_FILE)
    G.draw(OUTPUT_PNG_FILE, format="png", prog="dot")
    print(f"Flowchart generated: {OUTPUT_PNG_FILE}")

def main():
    # Analyze the codebase
    print("Analyzing codebase...")
    codebase_structure = analyze_codebase(REPO_PATH)
    
    # Summarize with LLM
    print("Summarizing codebase with LLM...")
    summary = summarize_codebase_with_llm(codebase_structure)
    print("\nCodebase Summary:")
    print(summary)
    
    # Generate flowchart
    print("\nGenerating flowchart...")
    generate_flowchart(codebase_structure)

if __name__ == "__main__":
    main()