#!/usr/bin/env python3
"""
Example usage of DocToAI library
"""

from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Example of using DocToAI programmatically."""
    
    # Import the main class
    from cli import DocToAI
    
    # Initialize with default configuration
    app = DocToAI()
    
    # Example 1: Process a single document
    print("=== Example 1: Single Document Processing ===")
    
    # Create a sample text file for testing
    sample_text = """
    # Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses on the development of algorithms 
    and statistical models that enable computer systems to improve their performance on a specific task 
    through experience.

    ## Types of Machine Learning

    ### Supervised Learning
    Supervised learning uses labeled training data to learn a mapping function from input variables to 
    output variables. Common examples include classification and regression problems.

    ### Unsupervised Learning  
    Unsupervised learning finds hidden patterns in data without labeled examples. Clustering and 
    dimensionality reduction are common unsupervised learning tasks.

    ### Reinforcement Learning
    Reinforcement learning is concerned with how agents should take actions in an environment to 
    maximize cumulative reward.

    ## Applications

    Machine learning has numerous applications across industries:
    - Healthcare: Medical diagnosis and drug discovery
    - Finance: Fraud detection and algorithmic trading  
    - Technology: Recommendation systems and natural language processing
    - Transportation: Autonomous vehicles and route optimization
    
    ## Conclusion
    
    As data continues to grow exponentially, machine learning will play an increasingly important role 
    in extracting insights and automating decision-making processes.
    """
    
    # Write sample file
    sample_file = Path("sample_document.md")
    with open(sample_file, 'w') as f:
        f.write(sample_text)
    
    try:
        # Process the document
        app.process_documents(
            input_paths=[sample_file],
            output_path=Path("example_output.jsonl"),
            mode="rag",
            chunk_strategy="semantic",
            output_format="jsonl",
            clean_text=True,
            verbose=True
        )
        
        print("✅ Successfully processed document!")
        print("📄 Output saved to: example_output.jsonl")
        
        # Read and display first few entries
        import json
        with open("example_output.jsonl", 'r') as f:
            entries = [json.loads(line) for line in f]
        
        print(f"\n📊 Generated {len(entries)} chunks")
        print("\n🔍 Sample chunks:")
        for i, entry in enumerate(entries[:2]):
            print(f"\nChunk {i+1}:")
            print(f"  ID: {entry['id']}")
            print(f"  Text: {entry['text'][:100]}...")
            print(f"  Metadata keys: {list(entry['metadata'].keys())}")
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()
    
    # Example 2: Fine-tuning format
    print("\n=== Example 2: Fine-tuning Format ===")
    
    # Create another sample file
    sample_file2 = Path("sample_qa.txt")
    with open(sample_file2, 'w') as f:
        f.write("""
        What is artificial intelligence?
        
        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
        that are programmed to think and learn like humans. AI systems can perform tasks that 
        typically require human intelligence, such as visual perception, speech recognition, 
        decision-making, and language translation.
        
        How does machine learning work?
        
        Machine learning works by using algorithms to analyze data, identify patterns, and make 
        predictions or decisions without being explicitly programmed for each specific task. 
        The system learns from training data and improves its performance over time as it 
        processes more information.
        """)
    
    try:
        app.process_documents(
            input_paths=[sample_file2],
            output_path=Path("finetune_output.jsonl"),
            mode="finetune",
            chunk_strategy="fixed", 
            output_format="jsonl",
            clean_text=True,
            verbose=True
        )
        
        print("✅ Generated fine-tuning dataset!")
        print("📄 Output saved to: finetune_output.jsonl")
        
        # Display sample
        with open("finetune_output.jsonl", 'r') as f:
            ft_entry = json.loads(f.readline())
        
        print("\n🔍 Sample fine-tuning entry:")
        print(f"  ID: {ft_entry['id']}")
        print(f"  Conversations: {len(ft_entry['conversations'])} turns")
        for turn in ft_entry['conversations']:
            print(f"    {turn['role']}: {turn['content'][:50]}...")
    
    finally:
        # Clean up
        if sample_file2.exists():
            sample_file2.unlink()
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("- Try processing your own documents")
    print("- Experiment with different chunking strategies")
    print("- Explore different output formats (JSON, CSV, Parquet)")
    print("- Use the CLI: doctoai convert your_document.pdf --output dataset.jsonl")


if __name__ == "__main__":
    main()