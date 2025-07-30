#!/usr/bin/env python3
"""
Data analysis and preparation utilities for the Qwen fine-tuning project.

This script provides tools for analyzing the reasoning traces dataset,
cleaning data, and preparing training splits.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Analyzes reasoning traces dataset for fine-tuning preparation."""
    
    def __init__(self, dataset_name: str = "microsoft/orca-math-word-problems-200k"):
        self.dataset_name = dataset_name
        self.output_dir = Path("results/data_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self) -> Dataset:
        """Load the dataset for analysis."""
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            dataset = load_dataset(self.dataset_name, split="train")
            logger.info(f"Loaded {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def analyze_basic_stats(self, dataset: Dataset) -> Dict:
        """Analyze basic dataset statistics."""
        logger.info("Analyzing basic dataset statistics...")
        
        stats = {
            "total_samples": len(dataset),
            "columns": list(dataset.features.keys()),
            "sample_example": dict(dataset[0]) if len(dataset) > 0 else {}
        }
        
        # Analyze text lengths
        text_fields = []
        for col in stats["columns"]:
            if dataset.features[col].dtype == "string":
                text_fields.append(col)
        
        for field in text_fields:
            texts = [item[field] for item in dataset if item[field]]
            lengths = [len(text) for text in texts]
            
            stats[f"{field}_stats"] = {
                "count": len(lengths),
                "mean_length": sum(lengths) / len(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "median_length": sorted(lengths)[len(lengths)//2] if lengths else 0
            }
        
        return stats
    
    def analyze_content_patterns(self, dataset: Dataset) -> Dict:
        """Analyze content patterns in the dataset."""
        logger.info("Analyzing content patterns...")
        
        patterns = {
            "programming_keywords": defaultdict(int),
            "mathematical_terms": defaultdict(int),
            "reasoning_indicators": defaultdict(int),
            "languages_detected": defaultdict(int)
        }
        
        # Define pattern lists
        prog_keywords = [
            "def ", "class ", "import ", "from ", "if ", "else:", "elif ",
            "for ", "while ", "try:", "except:", "return ", "print(",
            "input(", "len(", "range(", "list(", "dict(", "set("
        ]
        
        math_terms = [
            "equation", "solve", "calculate", "sum", "product", "average",
            "probability", "statistics", "algebra", "geometry", "calculus",
            "derivative", "integral", "matrix", "vector", "function"
        ]
        
        reasoning_indicators = [
            "step by step", "first", "then", "next", "finally", "therefore",
            "because", "since", "given that", "we can", "let's", "approach",
            "strategy", "solution", "answer", "result"
        ]
        
        # Analyze patterns
        for item in tqdm(dataset, desc="Analyzing patterns"):
            text_content = ""
            for field in item:
                if isinstance(item[field], str):
                    text_content += item[field].lower() + " "
            
            # Count programming keywords
            for keyword in prog_keywords:
                if keyword in text_content:
                    patterns["programming_keywords"][keyword] += 1
            
            # Count mathematical terms
            for term in math_terms:
                if term in text_content:
                    patterns["mathematical_terms"][term] += 1
            
            # Count reasoning indicators
            for indicator in reasoning_indicators:
                if indicator in text_content:
                    patterns["reasoning_indicators"][indicator] += 1
            
            # Detect potential programming languages
            if "python" in text_content:
                patterns["languages_detected"]["python"] += 1
            if "java" in text_content or "public class" in text_content:
                patterns["languages_detected"]["java"] += 1
            if "c++" in text_content or "#include" in text_content:
                patterns["languages_detected"]["cpp"] += 1
            if "javascript" in text_content or "function(" in text_content:
                patterns["languages_detected"]["javascript"] += 1
        
        # Convert to regular dicts and sort
        for category in patterns:
            patterns[category] = dict(sorted(
                patterns[category].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        return patterns
    
    def identify_quality_issues(self, dataset: Dataset) -> Dict:
        """Identify potential data quality issues."""
        logger.info("Identifying data quality issues...")
        
        issues = {
            "empty_fields": defaultdict(int),
            "too_short": defaultdict(int),
            "too_long": defaultdict(int),
            "duplicates": 0,
            "encoding_issues": 0,
            "incomplete_solutions": 0
        }
        
        seen_texts = set()
        
        for item in tqdm(dataset, desc="Checking quality"):
            for field, value in item.items():
                if isinstance(value, str):
                    # Check for empty fields
                    if not value.strip():
                        issues["empty_fields"][field] += 1
                    
                    # Check for too short content
                    if len(value.strip()) < 10:
                        issues["too_short"][field] += 1
                    
                    # Check for too long content
                    if len(value) > 10000:
                        issues["too_long"][field] += 1
                    
                    # Check for encoding issues
                    try:
                        value.encode('utf-8')
                    except UnicodeEncodeError:
                        issues["encoding_issues"] += 1
                    
                    # Check for duplicates (using a simple hash)
                    text_hash = hash(value.strip())
                    if text_hash in seen_texts:
                        issues["duplicates"] += 1
                    else:
                        seen_texts.add(text_hash)
            
            # Check for incomplete solutions (heuristic)
            text_content = " ".join([str(v) for v in item.values() if isinstance(v, str)])
            if "..." in text_content and len(text_content) < 100:
                issues["incomplete_solutions"] += 1
        
        return dict(issues)
    
    def create_cleaning_recommendations(self, stats: Dict, patterns: Dict, issues: Dict) -> Dict:
        """Create data cleaning recommendations based on analysis."""
        recommendations = {
            "filtering": [],
            "preprocessing": [],
            "quality_improvements": []
        }
        
        # Filtering recommendations
        if issues["too_short"]:
            recommendations["filtering"].append({
                "action": "Remove samples with very short content",
                "rationale": f"Found {sum(issues['too_short'].values())} samples with content < 10 characters",
                "threshold": "Min 50 characters per text field"
            })
        
        if issues["too_long"]:
            recommendations["filtering"].append({
                "action": "Truncate or remove very long samples",
                "rationale": f"Found {sum(issues['too_long'].values())} samples with content > 10k characters",
                "threshold": "Max 4096 characters per sample"
            })
        
        if issues["duplicates"] > 0:
            recommendations["filtering"].append({
                "action": "Remove duplicate samples",
                "rationale": f"Found {issues['duplicates']} potential duplicates",
                "method": "Hash-based deduplication"
            })
        
        # Preprocessing recommendations
        if issues["encoding_issues"] > 0:
            recommendations["preprocessing"].append({
                "action": "Fix encoding issues",
                "rationale": f"Found {issues['encoding_issues']} samples with encoding problems",
                "method": "Unicode normalization and cleanup"
            })
        
        # Focus on programming content if detected
        if patterns["programming_keywords"]:
            recommendations["preprocessing"].append({
                "action": "Enhance programming content formatting",
                "rationale": f"Found {len(patterns['programming_keywords'])} programming keywords",
                "method": "Standardize code block formatting"
            })
        
        # Quality improvements
        if issues["incomplete_solutions"] > 0:
            recommendations["quality_improvements"].append({
                "action": "Review incomplete solutions",
                "rationale": f"Found {issues['incomplete_solutions']} potentially incomplete solutions",
                "method": "Manual review or automated completion check"
            })
        
        return recommendations
    
    def save_analysis_results(self, stats: Dict, patterns: Dict, issues: Dict, recommendations: Dict):
        """Save analysis results to files."""
        logger.info("Saving analysis results...")
        
        # Save as JSON
        analysis_results = {
            "basic_stats": stats,
            "content_patterns": patterns,
            "quality_issues": issues,
            "cleaning_recommendations": recommendations,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(self.output_dir / "data_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Create summary report
        self.create_analysis_report(stats, patterns, issues, recommendations)
        
        logger.info(f"Analysis results saved to {self.output_dir}")
    
    def create_analysis_report(self, stats: Dict, patterns: Dict, issues: Dict, recommendations: Dict):
        """Create a human-readable analysis report."""
        report_lines = [
            "# Dataset Analysis Report",
            "",
            f"**Dataset:** {self.dataset_name}",
            f"**Total Samples:** {stats['total_samples']:,}",
            f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Basic Statistics",
            ""
        ]
        
        # Add basic stats
        for key, value in stats.items():
            if key.endswith("_stats") and isinstance(value, dict):
                field_name = key.replace("_stats", "")
                report_lines.extend([
                    f"### {field_name.title()} Field",
                    f"- Samples with content: {value['count']:,}",
                    f"- Average length: {value['mean_length']:.1f} characters",
                    f"- Length range: {value['min_length']} - {value['max_length']} characters",
                    ""
                ])
        
        # Add content patterns
        report_lines.extend([
            "## Content Analysis",
            ""
        ])
        
        for category, items in patterns.items():
            if items:
                report_lines.extend([
                    f"### {category.replace('_', ' ').title()}",
                    ""
                ])
                for item, count in list(items.items())[:10]:  # Top 10
                    report_lines.append(f"- {item}: {count:,} occurrences")
                report_lines.append("")
        
        # Add quality issues
        report_lines.extend([
            "## Quality Issues",
            ""
        ])
        
        for issue_type, count in issues.items():
            if isinstance(count, dict):
                if count:
                    report_lines.append(f"### {issue_type.replace('_', ' ').title()}")
                    for field, field_count in count.items():
                        report_lines.append(f"- {field}: {field_count:,} issues")
                    report_lines.append("")
            else:
                if count > 0:
                    report_lines.append(f"- {issue_type.replace('_', ' ').title()}: {count:,}")
        
        # Add recommendations
        report_lines.extend([
            "",
            "## Cleaning Recommendations",
            ""
        ])
        
        for category, recs in recommendations.items():
            if recs:
                report_lines.extend([
                    f"### {category.replace('_', ' ').title()}",
                    ""
                ])
                for i, rec in enumerate(recs, 1):
                    report_lines.extend([
                        f"{i}. **{rec['action']}**",
                        f"   - Rationale: {rec['rationale']}",
                        f"   - Implementation: {rec.get('method', rec.get('threshold', 'Manual review'))}",
                        ""
                    ])
        
        # Save report
        with open(self.output_dir / "analysis_report.md", 'w') as f:
            f.write("\n".join(report_lines))
    
    def run_full_analysis(self):
        """Run complete dataset analysis."""
        logger.info("Starting comprehensive dataset analysis...")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Run analyses
        stats = self.analyze_basic_stats(dataset)
        patterns = self.analyze_content_patterns(dataset)
        issues = self.identify_quality_issues(dataset)
        recommendations = self.create_cleaning_recommendations(stats, patterns, issues)
        
        # Save results
        self.save_analysis_results(stats, patterns, issues, recommendations)
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET ANALYSIS SUMMARY")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Columns: {', '.join(stats['columns'])}")
        
        if patterns["programming_keywords"]:
            print(f"Programming content detected: {len(patterns['programming_keywords'])} keywords found")
        
        if patterns["mathematical_terms"]:
            print(f"Mathematical content detected: {len(patterns['mathematical_terms'])} terms found")
        
        total_issues = sum([
            sum(issues['empty_fields'].values()) if isinstance(issues['empty_fields'], dict) else issues['empty_fields'],
            sum(issues['too_short'].values()) if isinstance(issues['too_short'], dict) else issues['too_short'],
            sum(issues['too_long'].values()) if isinstance(issues['too_long'], dict) else issues['too_long'],
            issues['duplicates'],
            issues['encoding_issues'],
            issues['incomplete_solutions']
        ])
        
        print(f"Quality issues found: {total_issues:,}")
        print(f"Recommendations generated: {sum(len(recs) for recs in recommendations.values())}")
        print(f"\nDetailed results saved to: {self.output_dir}")
        print("="*60)
        
        return {
            "stats": stats,
            "patterns": patterns,
            "issues": issues,
            "recommendations": recommendations
        }

def main():
    """Main function."""
    analyzer = DataAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\nAnalysis completed successfully!")
    print("Check the results directory for detailed reports.")

if __name__ == "__main__":
    main()