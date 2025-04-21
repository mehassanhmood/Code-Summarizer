import os
import ast
import argparse
import logging
import json
import pygraphviz as pgv
from pathlib import Path
import ollama
from functools import lru_cache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CodeHelper")

class CodeHelper:
    def __init__(self, repo_path, output_dir=None, model_name="codellama", log_level="INFO"):
        """Initialize the CodeHelper with configuration parameters."""
        self.repo_path = os.path.abspath(repo_path)
        
        # Create output directory inside the repo by default
        if output_dir is None:
            self.output_dir = os.path.join(self.repo_path, "code_analysis")
        else:
            self.output_dir = output_dir
            
        self.model_name = model_name
        
        # Set up output directory
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.output_dot_file = os.path.join(self.output_dir, "codebase_flowchart.dot")
        self.output_png_file = os.path.join(self.output_dir, "codebase_flowchart.png")
        self.output_json_file = os.path.join(self.output_dir, "codebase_structure.json")
        self.output_summary_file = os.path.join(self.output_dir, "codebase_summary.txt")
        
        # Create a log file in the output directory
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "code_analysis.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Set log level
        logger.setLevel(getattr(logging, log_level))
        logger.info(f"Analysis output will be saved to: {self.output_dir}")

    @lru_cache(maxsize=100)
    def parse_python_file(self, file_path):
        """Parse a Python file and extract classes, methods, functions, docstrings, and imports."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                source_code = file.read()
                tree = ast.parse(source_code, filename=file_path)
            
            classes = []
            functions = []
            imports = []
            docstring = ast.get_docstring(tree)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        imports.append(f"{module}.{name.name}")
            
            # Extract classes and their methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    class_docstring = ast.get_docstring(node)
                    
                    for method_node in node.body:
                        if isinstance(method_node, ast.FunctionDef):
                            method_docstring = ast.get_docstring(method_node)
                            methods.append({
                                "name": method_node.name,
                                "docstring": method_docstring or "",
                                "arguments": [arg.arg for arg in method_node.args.args if arg.arg != "self"]
                            })
                    
                    classes.append({
                        "name": node.name,
                        "methods": methods,
                        "docstring": class_docstring or ""
                    })
                elif isinstance(node, ast.FunctionDef) and not isinstance(node.parent if hasattr(node, "parent") else None, ast.ClassDef):
                    function_docstring = ast.get_docstring(node)
                    functions.append({
                        "name": node.name,
                        "docstring": function_docstring or "",
                        "arguments": [arg.arg for arg in node.args.args]
                    })
            
            # Extract simple code metrics
            code_quality = {
                "loc": len(source_code.splitlines()),
                "num_classes": len(classes),
                "num_functions": len(functions),
                "has_docstrings": bool(docstring) or any(cls["docstring"] for cls in classes)
            }
            
            return {
                "file": file_path,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "docstring": docstring or "",
                "metrics": code_quality
            }
        except SyntaxError as e:
            logger.error(f"Syntax error parsing {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def analyze_codebase(self):
        """Walk through the repository and analyze all Python files."""
        logger.info(f"Analyzing codebase at {self.repo_path}")
        codebase_structure = {}
        file_count = 0
        
        # Make sure we don't analyze our output directory
        output_rel_path = os.path.relpath(self.output_dir, self.repo_path)
        
        for root, _, files in os.walk(self.repo_path):
            # Skip our output directory, virtual environments and hidden directories
            rel_path = os.path.relpath(root, self.repo_path)
            if (rel_path.startswith(output_rel_path) or 
                any(part.startswith('.') or part == '__pycache__' or part == 'venv' 
                   for part in Path(rel_path).parts)):
                continue
                
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_path)
                    parsed_data = self.parse_python_file(file_path)
                    if parsed_data:
                        codebase_structure[relative_path] = parsed_data
                        file_count += 1
                        if file_count % 10 == 0:
                            logger.info(f"Analyzed {file_count} Python files...")
        
        logger.info(f"Completed analysis of {file_count} Python files")
        return codebase_structure

    def find_dependencies(self, codebase_structure):
        """Find dependencies between modules based on imports."""
        dependencies = {}
        module_paths = {}
        
        # Map module names to file paths
        for file_path, data in codebase_structure.items():
            module_name = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
            module_paths[module_name] = file_path
        
        # Find dependencies
        for file_path, data in codebase_structure.items():
            dependencies[file_path] = []
            module_name = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
            
            for imported in data.get("imports", []):
                # Check if this import refers to another module in our codebase
                for potential_module in module_paths:
                    if imported == potential_module or imported.startswith(potential_module + "."):
                        dependencies[file_path].append(module_paths[potential_module])
                        break
        
        return dependencies

    def summarize_codebase_with_llm(self, codebase_structure):
        """Use a local LLM to summarize the codebase and infer what needs to be done."""
        logger.info(f"Summarizing codebase with {self.model_name}")
        
        summary_prompt = """
You are an AI assistant analyzing a Python codebase for an investment fund using ML and AI. Below is the structure of the codebase, including modules, classes, methods, and functions. Provide:
1. A detailed summary of the codebase's purpose and functionality.
2. Sub Categorized the Classes or metods if need be and thier purpose.
2. Insights into what has been implemented.
3. Suggestions for what needs to be done next (e.g., missing components, optimizations).
4. Identify any potential software architecture issues, anti-patterns, or areas for refactoring.

Codebase structure:
"""
        for module, data in codebase_structure.items():
            summary_prompt += f"\nModule: {module}\n"
            if data["docstring"]:
                summary_prompt += f"Module Docstring: {data['docstring'][:200]}...\n"
            
            if data["classes"]:
                summary_prompt += "Classes:\n"
                for cls in data["classes"]:
                    method_names = [m["name"] for m in cls["methods"]]
                    summary_prompt += f"  - {cls['name']} (Methods: {', '.join(method_names)})\n"
                    if cls["docstring"]:
                        summary_prompt += f"    Docstring: {cls['docstring'][:100]}...\n"
            
            if data["functions"]:
                summary_prompt += "Functions:\n"
                for func in data["functions"]:
                    summary_prompt += f"  - {func['name']}({', '.join(func['arguments'])})\n"
                    if func["docstring"]:
                        summary_prompt += f"    Docstring: {func['docstring'][:100]}...\n"
            
            if data["imports"]:
                summary_prompt += f"Imports: {', '.join(data['imports'][:10])}\n"
                if len(data["imports"]) > 10:
                    summary_prompt += f"...and {len(data['imports']) - 10} more imports\n"
        
        try:
            response = ollama.generate(model=self.model_name, prompt=summary_prompt)
            summary = response["response"]
            
            # Save the summary to a file
            with open(self.output_summary_file, "w", encoding="utf-8") as f:
                f.write(summary)
            
            logger.info(f"Saved codebase summary to {self.output_summary_file}")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing with LLM: {e}")
            return f"Error summarizing with LLM: {e}"

    def generate_flowchart(self, codebase_structure, dependencies):
        """Generate a flowchart of the codebase using Graphviz."""
        logger.info("Generating codebase flowchart")
        G = pgv.AGraph(directed=True, rankdir="LR", overlap="scale", splines="true", 
                        name="Codebase Flowchart", bgcolor="white", fontname="Arial")
        
        # Define node styles
        G.node_attr.update(fontsize="10", margin="0.1,0.1", height="0.2")
        G.edge_attr.update(fontsize="8")
        
        # Add nodes for each module
        for module, data in codebase_structure.items():
            module_node = module.replace("/", "_").replace(".", "_").replace("\\", "_")
            
            # Calculate module complexity
            num_classes = len(data["classes"])
            num_methods = sum(len(cls["methods"]) for cls in data["classes"])
            num_functions = len(data["functions"])
            complexity = num_classes + num_methods + num_functions
            
            # Set node color based on complexity
            color = "lightblue"
            if complexity > 20:
                color = "red"
            elif complexity > 10:
                color = "orange"
            elif complexity > 5:
                color = "yellow"
            
            G.add_node(module_node, label=f"{module}\n({num_classes} classes, {num_functions} funcs)", 
                      shape="box", style="filled", fillcolor=color)
            
            # Add class nodes
            for cls in data["classes"]:
                class_node = f"{module_node}_{cls['name']}"
                method_list = ", ".join([m["name"] for m in cls["methods"]][:5])
                if len(cls["methods"]) > 5:
                    method_list += "..."
                    
                G.add_node(class_node, label=f"{cls['name']}\n{method_list}", 
                          shape="ellipse", style="filled", fillcolor="lightgreen")
                G.add_edge(module_node, class_node)
        
        # Add dependency edges
        for module, deps in dependencies.items():
            module_node = module.replace("/", "_").replace(".", "_").replace("\\", "_")
            for dep in deps:
                dep_node = dep.replace("/", "_").replace(".", "_").replace("\\", "_")
                G.add_edge(module_node, dep_node, style="dashed", color="gray", penwidth="0.5")
        
        # Save the flowchart
        try:
            G.write(self.output_dot_file)
            G.draw(self.output_png_file, format="png", prog="dot")
            logger.info(f"Flowchart generated: {self.output_png_file}")
        except Exception as e:
            logger.error(f"Error generating flowchart: {e}")

    def export_codebase_structure(self, codebase_structure):
        """Export the codebase structure to a JSON file."""
        try:
            with open(self.output_json_file, "w", encoding="utf-8") as f:
                json.dump(codebase_structure, f, indent=2)
            logger.info(f"Exported codebase structure to {self.output_json_file}")
        except Exception as e:
            logger.error(f"Error exporting codebase structure: {e}")

    def run(self):
        """Run the complete analysis pipeline."""
        # Analyze the codebase
        codebase_structure = self.analyze_codebase()
        
        # Find dependencies
        dependencies = self.find_dependencies(codebase_structure)
        
        # Export structure to JSON
        self.export_codebase_structure(codebase_structure)
        
        # Summarize with LLM
        summary = self.summarize_codebase_with_llm(codebase_structure)
        print("\nCodebase Summary:")
        print(summary)
        
        # Generate flowchart
        self.generate_flowchart(codebase_structure, dependencies)
        
        logger.info(f"Analysis complete. All outputs saved to {self.output_dir}")
        
        return {
            "structure": codebase_structure,
            "dependencies": dependencies,
            "summary": summary
        }


def main():
    """Parse command line arguments and run the code helper."""
    parser = argparse.ArgumentParser(description="Python Codebase Analysis Tool")
    parser.add_argument("--repo", "-r", default=".", help="Repository path to analyze")
    parser.add_argument("--output", "-o", default=None, 
                        help="Output directory for results (defaults to 'code_analysis' inside the repo)")
    parser.add_argument("--model", "-m", default="codellama", help="Ollama model for code understanding")
    parser.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    try:
        helper = CodeHelper(
            repo_path=args.repo,
            output_dir=args.output,
            model_name=args.model,
            log_level=args.log_level
        )
        helper.run()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Error running analysis: {e}", exc_info=True)

if __name__ == "__main__":
    main()
