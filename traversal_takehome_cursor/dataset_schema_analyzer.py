#!/usr/bin/env python
"""
Dataset Schema Analyzer for open-r1/Mixture-of-Thoughts

Generates detailed schema documentation and exports structure to JSON/CSV formats.
"""

import json
import csv
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict, Counter
import statistics
from datetime import datetime

class DatasetSchemaAnalyzer:
    def __init__(self, dataset_name="open-r1/Mixture-of-Thoughts", config="all"):
        self.dataset_name = dataset_name
        self.config = config
        self.analysis_results = {}
        
    def load_dataset_sample(self, sample_size=1000):
        """Load a sample of the dataset for analysis"""
        print(f"Loading {sample_size} samples from {self.dataset_name}...")
        
        try:
            # Try to load a sample
            self.dataset = load_dataset(
                self.dataset_name, 
                self.config, 
                split=f"train[:{sample_size}]"
            )
            print(f"✅ Loaded {len(self.dataset)} examples")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def analyze_schema(self):
        """Analyze the complete schema of the dataset"""
        print("\n🔍 Analyzing dataset schema...")
        
        schema_info = {
            "dataset_name": self.dataset_name,
            "config": self.config,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_examples": len(self.dataset),
            "columns": {},
            "message_schema": {},
            "statistics": {}
        }
        
        # Analyze each column
        for column_name in self.dataset.column_names:
            schema_info["columns"][column_name] = self._analyze_column(column_name)
        
        # Special analysis for messages column if it exists
        if "messages" in self.dataset.column_names:
            schema_info["message_schema"] = self._analyze_messages_schema()
        
        # General statistics
        schema_info["statistics"] = self._compute_statistics()
        
        self.analysis_results = schema_info
        return schema_info
    
    def _analyze_column(self, column_name):
        """Analyze a specific column"""
        column_data = [example[column_name] for example in self.dataset]
        
        analysis = {
            "type": str(self.dataset.features[column_name]),
            "non_null_count": sum(1 for x in column_data if x is not None),
            "null_count": sum(1 for x in column_data if x is None),
            "unique_count": len(set(str(x) for x in column_data)),
            "sample_values": []
        }
        
        # Add sample values (first 5 unique ones)
        unique_values = list(set(str(x) for x in column_data))[:5]
        analysis["sample_values"] = unique_values
        
        # Type-specific analysis
        if column_name == "messages":
            analysis.update(self._analyze_messages_column(column_data))
        elif all(isinstance(x, str) for x in column_data if x is not None):
            analysis.update(self._analyze_string_column(column_data))
        elif all(isinstance(x, (int, float)) for x in column_data if x is not None):
            analysis.update(self._analyze_numeric_column(column_data))
        
        return analysis
    
    def _analyze_string_column(self, data):
        """Analyze string column"""
        lengths = [len(str(x)) for x in data if x is not None]
        return {
            "avg_length": statistics.mean(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "median_length": statistics.median(lengths) if lengths else 0
        }
    
    def _analyze_numeric_column(self, data):
        """Analyze numeric column"""
        numbers = [x for x in data if x is not None and isinstance(x, (int, float))]
        if not numbers:
            return {}
        
        return {
            "min_value": min(numbers),
            "max_value": max(numbers),
            "mean_value": statistics.mean(numbers),
            "median_value": statistics.median(numbers),
            "std_dev": statistics.stdev(numbers) if len(numbers) > 1 else 0
        }
    
    def _analyze_messages_column(self, data):
        """Analyze the messages column specifically"""
        conversation_lengths = []
        total_messages = 0
        
        for conversation in data:
            if isinstance(conversation, list):
                conversation_lengths.append(len(conversation))
                total_messages += len(conversation)
        
        return {
            "total_conversations": len(conversation_lengths),
            "total_messages": total_messages,
            "avg_messages_per_conversation": statistics.mean(conversation_lengths) if conversation_lengths else 0,
            "min_messages": min(conversation_lengths) if conversation_lengths else 0,
            "max_messages": max(conversation_lengths) if conversation_lengths else 0,
            "median_messages": statistics.median(conversation_lengths) if conversation_lengths else 0
        }
    
    def _analyze_messages_schema(self):
        """Analyze the schema of individual messages"""
        print("  📋 Analyzing message structure...")
        
        all_message_keys = set()
        role_counts = Counter()
        content_lengths = []
        message_types = Counter()
        
        for example in self.dataset:
            messages = example.get("messages", [])
            
            for msg in messages:
                if isinstance(msg, dict):
                    # Collect all possible keys in messages
                    all_message_keys.update(msg.keys())
                    
                    # Analyze roles
                    role = msg.get("role", "unknown")
                    role_counts[role] += 1
                    
                    # Analyze content
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        content_lengths.append(len(content))
                        
                        # Classify message types
                        if "```" in content:
                            message_types["code"] += 1
                        elif any(keyword in content.lower() for keyword in ["math", "equation", "calculate"]):
                            message_types["math"] += 1
                        elif len(content.split()) > 50:
                            message_types["long_text"] += 1
                        else:
                            message_types["regular"] += 1
        
        return {
            "message_keys": list(all_message_keys),
            "role_distribution": dict(role_counts),
            "message_type_distribution": dict(message_types),
            "content_length_stats": {
                "avg_length": statistics.mean(content_lengths) if content_lengths else 0,
                "min_length": min(content_lengths) if content_lengths else 0,
                "max_length": max(content_lengths) if content_lengths else 0,
                "median_length": statistics.median(content_lengths) if content_lengths else 0
            }
        }
    
    def _compute_statistics(self):
        """Compute general dataset statistics"""
        return {
            "total_examples": len(self.dataset),
            "total_columns": len(self.dataset.column_names),
            "column_names": self.dataset.column_names,
            "dataset_features": {name: str(feature) for name, feature in self.dataset.features.items()}
        }
    
    def export_schema(self, output_dir="dataset_analysis"):
        """Export schema analysis to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export to JSON
        json_file = output_path / "mixture_of_thoughts_schema.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Schema exported to {json_file}")
        
        # Export summary to CSV
        csv_file = output_path / "mixture_of_thoughts_summary.csv"
        self._export_csv_summary(csv_file)
        print(f"✅ Summary exported to {csv_file}")
        
        # Create markdown report
        md_file = output_path / "mixture_of_thoughts_report.md"
        self._create_markdown_report(md_file)
        print(f"✅ Report exported to {md_file}")
    
    def _export_csv_summary(self, csv_file):
        """Export a CSV summary"""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            
            # Basic info
            writer.writerow(["Dataset", self.dataset_name])
            writer.writerow(["Config", self.config])
            writer.writerow(["Total Examples", self.analysis_results["total_examples"]])
            writer.writerow(["Total Columns", len(self.analysis_results["columns"])])
            
            # Column info
            for col_name, col_info in self.analysis_results["columns"].items():
                writer.writerow([f"Column: {col_name}", col_info["type"]])
                if "avg_length" in col_info:
                    writer.writerow([f"{col_name} - Avg Length", f"{col_info['avg_length']:.2f}"])
    
    def _create_markdown_report(self, md_file):
        """Create a detailed markdown report"""
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Dataset Analysis Report: {self.dataset_name}\n\n")
            f.write(f"**Analysis Date:** {self.analysis_results['analysis_timestamp']}\n\n")
            f.write(f"**Configuration:** {self.config}\n\n")
            
            # Basic Statistics
            f.write("## Basic Statistics\n\n")
            stats = self.analysis_results["statistics"]
            f.write(f"- **Total Examples:** {stats['total_examples']:,}\n")
            f.write(f"- **Total Columns:** {stats['total_columns']}\n")
            f.write(f"- **Column Names:** {', '.join(stats['column_names'])}\n\n")
            
            # Column Analysis
            f.write("## Column Analysis\n\n")
            for col_name, col_info in self.analysis_results["columns"].items():
                f.write(f"### {col_name}\n\n")
                f.write(f"- **Type:** {col_info['type']}\n")
                f.write(f"- **Non-null Count:** {col_info['non_null_count']:,}\n")
                f.write(f"- **Unique Values:** {col_info['unique_count']:,}\n")
                
                if "avg_length" in col_info:
                    f.write(f"- **Average Length:** {col_info['avg_length']:.2f}\n")
                    f.write(f"- **Length Range:** {col_info['min_length']} - {col_info['max_length']}\n")
                
                f.write("\n")
            
            # Message Schema (if exists)
            if "message_schema" in self.analysis_results:
                f.write("## Message Schema Analysis\n\n")
                msg_schema = self.analysis_results["message_schema"]
                
                f.write(f"- **Message Keys:** {', '.join(msg_schema['message_keys'])}\n")
                f.write(f"- **Role Distribution:**\n")
                for role, count in msg_schema['role_distribution'].items():
                    f.write(f"  - {role}: {count:,}\n")
                
                content_stats = msg_schema['content_length_stats']
                f.write(f"- **Content Length Stats:**\n")
                f.write(f"  - Average: {content_stats['avg_length']:.2f}\n")
                f.write(f"  - Range: {content_stats['min_length']} - {content_stats['max_length']}\n")
    
    def print_summary(self):
        """Print a summary to console"""
        print("\n" + "="*60)
        print("📊 DATASET SCHEMA SUMMARY")
        print("="*60)
        
        print(f"Dataset: {self.dataset_name}")
        print(f"Configuration: {self.config}")
        print(f"Total Examples: {self.analysis_results['total_examples']:,}")
        print(f"Columns: {len(self.analysis_results['columns'])}")
        
        print(f"\nColumn Details:")
        for col_name, col_info in self.analysis_results['columns'].items():
            print(f"  {col_name}: {col_info['type']}")
            if 'avg_messages_per_conversation' in col_info:
                print(f"    Avg messages/conversation: {col_info['avg_messages_per_conversation']:.1f}")

def main():
    """Main function"""
    print("🚀 Starting Mixture-of-Thoughts Dataset Schema Analysis")
    
    analyzer = DatasetSchemaAnalyzer()
    
    # Load dataset
    if not analyzer.load_dataset_sample(sample_size=1000):
        return 1
    
    # Analyze schema
    analyzer.analyze_schema()
    
    # Print summary
    analyzer.print_summary()
    
    # Export results
    analyzer.export_schema()
    
    print("\n✅ Analysis complete! Check the 'dataset_analysis' folder for detailed reports.")
    return 0

if __name__ == "__main__":
    exit(main())