import torch
import os
import json
import argparse
import random
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

class AVDNNonsenseReplacer:
    """Replace AVDN instructions with random nonsense instructions while preserving structure."""
    
    def __init__(self, nonsense_file: str, output_dir: str):
        self.output_dir = output_dir
        self.nonsense_instructions = self.load_nonsense_instructions(nonsense_file)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize usage tracking
        self.instruction_usage = {instruction: 0 for instruction in self.nonsense_instructions}
        
        print(f"Loaded {len(self.nonsense_instructions)} nonsense instructions")
    
    def load_nonsense_instructions(self, nonsense_file: str) -> List[str]:
        """Load nonsense instructions from JSON file."""
        with open(nonsense_file, 'r') as f:
            data = json.load(f)
        
        # Extract just the instruction strings
        instructions = [item['nonsense_instruction'] for item in data]
        return instructions
    
    def load_avdn_data(self, split: str) -> List[Dict]:
        """Load original AVDN dataset for a specific split."""
        # Use the same path structure as the original script
        data_file = f"../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/{split}_data.json"
        print(f"Loading AVDN data from: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples from {split} split")
        return data
    
    def find_similar_length_nonsense(self, original_instruction: str, split: str) -> str:
        """Find a nonsense instruction with similar length to the original, respecting reuse constraints."""
        original_length = len(original_instruction)
        
        # Set reuse constraints based on split
        max_reuse = 5 if split == 'train' else 2
        
        # Find instructions within 20% length difference
        min_length = int(original_length * 0.8)
        max_length = int(original_length * 1.2)
        
        candidates = []
        for instruction in self.nonsense_instructions:
            if (min_length <= len(instruction) <= max_length and 
                self.instruction_usage[instruction] < max_reuse):
                candidates.append(instruction)
        
        if candidates:
            selected = random.choice(candidates)
            self.instruction_usage[selected] += 1
            return selected
        else:
            # If no candidates found within constraints, pick any available instruction
            available_instructions = [inst for inst in self.nonsense_instructions 
                                   if self.instruction_usage[inst] < max_reuse]
            
            if available_instructions:
                selected = random.choice(available_instructions)
                self.instruction_usage[selected] += 1
                return selected
            else:
                # If all instructions are at max usage, pick randomly
                selected = random.choice(self.nonsense_instructions)
                self.instruction_usage[selected] += 1
                return selected
    
    def update_avdn_instruction(self, avdn_sample: Dict, nonsense_instruction: str, turn_index: int) -> Dict:
        """Update AVDN sample with nonsense instruction and preserve dialog structure."""
        # Create new sample by copying the original
        new_sample = avdn_sample.copy()
        
        # Get the original instruction
        original_instruction = avdn_sample['instructions']
        
        # Handle different instruction formats
        if '[QUE]' in original_instruction and '[INS]' in original_instruction:
            # Complex format with question and answer
            parts = original_instruction.split('[INS]')
            question_part = parts[0].replace('[QUE]', '').strip()
            
            # Create new instruction with nonsense answer
            new_instruction = f"[QUE] {question_part} [INS] {nonsense_instruction}"
        else:
            # Simple instruction format - replace the instruction part
            if '[INS]' in original_instruction:
                # Keep the [INS] tag and replace the content
                new_instruction = f"[INS] {nonsense_instruction}"
            else:
                # No [INS] tag, just replace the whole instruction
                new_instruction = nonsense_instruction
        
        # Update the instruction
        new_sample['instructions'] = new_instruction
        
        # Add debug information for verification
        new_sample['_debug_info'] = {
            'original_instruction': original_instruction,
            'nonsense_instruction': nonsense_instruction,
            'original_length': len(original_instruction),
            'nonsense_length': len(nonsense_instruction),
            'length_ratio': len(nonsense_instruction) / len(original_instruction) if len(original_instruction) > 0 else 0
        }
        
        return new_sample
    
    def process_avdn_sample(self, avdn_sample: Dict, avdn_index: int, split: str) -> Dict:
        """Process a single AVDN sample and replace with nonsense instruction."""
        
        # Get AVDN sample metadata
        map_name = avdn_sample['map_name']
        route_index = avdn_sample['route_index']
        
        # Extract original instruction
        original_instruction = avdn_sample['instructions']
        
        # Only process Q&A entries (skip non-Q&A samples)
        if '[QUE]' not in original_instruction or '[INS]' not in original_instruction:
            if avdn_index < 10:  # Only log first few skipped samples
                print(f"⚠️  Skipping non-Q&A sample {avdn_index} ({map_name}_{route_index})")
            return avdn_sample  # Return original sample unchanged
        
        # Extract the answer part for length matching
        ins_start = original_instruction.find('[INS]')
        if ins_start != -1:
            original_answer = original_instruction[ins_start+5:].strip()
        else:
            original_answer = original_instruction.strip()
        
        # Find similar length nonsense instruction
        nonsense_instruction = self.find_similar_length_nonsense(original_answer, split)
        
        # Update AVDN sample with nonsense instruction
        new_sample = self.update_avdn_instruction(avdn_sample, nonsense_instruction, 0)
        
        # Debug output for first few samples
        if avdn_index < 3:
            print(f"\n🔍 Nonsense Replacement Debug for sample {avdn_index}:")
            print(f"   Map: {map_name}, Route: {route_index}")
            print(f"   Original: {original_answer}")
            print(f"   Nonsense: {nonsense_instruction}")
            print(f"   Length ratio: {len(nonsense_instruction)/len(original_answer):.2f}")
            print(f"   Usage count: {self.instruction_usage[nonsense_instruction]}")
        
        return new_sample
    
    def update_pre_dialogs(self, data: List[Dict]) -> List[Dict]:
        """Update pre_dialogs for all samples based on nonsense instructions."""
        print("🔄 Starting pre_dialogs update with nonsense instructions...")
        
        # Group samples by episode for processing
        episodes = {}
        for i, sample in enumerate(data):
            map_name = sample['map_name']
            route_index = sample['route_index']
            
            episode_key = route_index.rsplit('_', 1)[0]  # Remove turn number
            full_episode_key = f"{map_name}_{episode_key}"
            
            if full_episode_key not in episodes:
                episodes[full_episode_key] = []
            episodes[full_episode_key].append((i, sample))
        
        # Sort episodes by turn number for proper processing order
        for episode_key in episodes:
            episodes[episode_key].sort(key=lambda x: int(x[1]['route_index'].split('_')[-1]))
        
        print(f"📊 Found {len(episodes)} episodes to process")
        
        # Create copy of data for updates
        updated_data = [sample.copy() for sample in data]
        
        # Process each episode
        for episode_key, episode_samples in episodes.items():
            if len(episode_samples) > 1:  # Only process episodes with multiple turns
                print(f"🔄 Processing episode {episode_key} with {len(episode_samples)} turns")
                
                # Process each turn in the episode
                for turn_idx, (sample_idx, sample) in enumerate(episode_samples):
                    if turn_idx > 0:  # Skip the first turn as it has no previous instructions
                        # Get the instruction from the current turn
                        new_instruction = sample['instructions']
                        
                        # Remove debug info before updating pre_dialogs
                        if '_debug_info' in updated_data[sample_idx]:
                            updated_data[sample_idx].pop('_debug_info')
                        
                        # Update pre_dialogs for all subsequent turns in this episode
                        for future_turn_idx in range(turn_idx+1, len(episode_samples)):
                            future_sample_idx = episode_samples[future_turn_idx][0]  # Get the actual array index
                            # Update the pre_dialogs at the correct position
                            if turn_idx < len(updated_data[future_sample_idx]['pre_dialogs']):
                                updated_data[future_sample_idx]['pre_dialogs'][turn_idx] = new_instruction
        
        return updated_data
    
    def process_split(self, split: str, sample_ratio: float = 1.0) -> List[Dict]:
        """Process an entire split with nonsense instruction replacement."""
        print(f"\n🚀 Processing {split} split with nonsense instructions...")
        
        # Load AVDN data
        avdn_data = self.load_avdn_data(split)
        
        # Apply sampling if needed
        if sample_ratio < 1.0:
            num_samples = int(len(avdn_data) * sample_ratio)
            avdn_data = avdn_data[:num_samples]
            print(f"📊 Sampled {num_samples}/{len(self.load_avdn_data(split))} samples ({sample_ratio*100:.1f}%)")
        
        # Process samples
        processed_data = []
        successful_replacements = 0
        total_qa_samples = 0
        
        for i, sample in enumerate(avdn_data):
            try:
                processed_sample = self.process_avdn_sample(sample, i, split)
                processed_data.append(processed_sample)
                
                # Check if replacement was successful
                if processed_sample['instructions'] != sample['instructions']:
                    successful_replacements += 1
                
                # Count Q&A samples
                if '[QUE]' in sample['instructions'] and '[INS]' in sample['instructions']:
                    total_qa_samples += 1
                
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                processed_data.append(sample)
        
        # Update pre_dialogs
        processed_data = self.update_pre_dialogs(processed_data)
        
        # Calculate usage statistics
        used_instructions = [inst for inst, count in self.instruction_usage.items() if count > 0]
        total_usage = sum(self.instruction_usage.values())
        avg_usage = total_usage / len(used_instructions) if used_instructions else 0
        max_usage = max(self.instruction_usage.values()) if self.instruction_usage else 0
        min_usage = min(self.instruction_usage.values()) if self.instruction_usage else 0
        
        # Print statistics
        print(f"\n📊 {split.upper()} NONSENSE REPLACEMENT STATISTICS:")
        print(f"Total Samples: {len(processed_data)}")
        print(f"Q&A Samples: {total_qa_samples}")
        print(f"Successful Replacements: {successful_replacements}")
        print(f"Success Rate: {successful_replacements/total_qa_samples*100:.1f}%" if total_qa_samples > 0 else "No Q&A samples found")
        print(f"\n📈 NONSENSE INSTRUCTION USAGE STATISTICS:")
        print(f"Total Instructions Used: {len(used_instructions)}/{len(self.nonsense_instructions)}")
        print(f"Average Usage per Instruction: {avg_usage:.2f}")
        print(f"Maximum Usage: {max_usage}")
        print(f"Minimum Usage: {min_usage}")
        print(f"Total Usage Count: {total_usage}")
        
        # Print examples
        examples_shown = 0
        for i, (original, processed) in enumerate(zip(avdn_data, processed_data)):
            if examples_shown >= 3:
                break
            if processed['instructions'] != original['instructions']:
                examples_shown += 1
                print(f"\nGenerated Example {examples_shown}:")
                print(f"Map: {original['map_name']}, Route: {original['route_index']}")
                print(f"Original: {original['instructions']}")
                print(f"Nonsense: {processed['instructions']}")
                if '_debug_info' in processed:
                    debug = processed['_debug_info']
                    print(f"Length ratio: {debug['length_ratio']:.2f}")
                print("-" * 80)
        
        return processed_data, {
            'total_samples': len(processed_data),
            'qa_samples': total_qa_samples,
            'successful_replacements': successful_replacements,
            'success_rate': successful_replacements / total_qa_samples if total_qa_samples > 0 else 0.0,
            'usage_stats': {
                'used_instructions': len(used_instructions),
                'total_instructions': len(self.nonsense_instructions),
                'avg_usage': avg_usage,
                'max_usage': max_usage,
                'min_usage': min_usage,
                'total_usage': total_usage
            }
        }
    
    def save_processed_data(self, data: List[Dict], split: str):
        """Save processed data to output directory in AVDN format."""
        # Save main AVDN dataset
        output_file = os.path.join(self.output_dir, f'{split}_data.json')
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} samples to {output_file}")
    
    def process_all_splits(self, splits: List[str], sample_ratio: float = 1.0):
        """Process all specified splits with nonsense instruction replacement."""
        overall_metrics = {}
        
        for split in splits:
            processed_data, split_metrics = self.process_split(split, sample_ratio)
            
            # Save processed data
            self.save_processed_data(processed_data, split)
            
            # Store metrics
            overall_metrics[split] = split_metrics
        
        # Print overall summary
        print(f"\n🎯 OVERALL NONSENSE REPLACEMENT SUMMARY:")
        print("=" * 80)
        for split, metrics in overall_metrics.items():
            print(f"{split.upper()}:")
            print(f"  📊 Replacement: {metrics['successful_replacements']}/{metrics['qa_samples']} Q&A samples replaced ({metrics['success_rate']*100:.1f}% success)")
            print(f"  📈 Usage: {metrics['usage_stats']['used_instructions']}/{metrics['usage_stats']['total_instructions']} instructions used")
            print(f"  📊 Average usage: {metrics['usage_stats']['avg_usage']:.2f} per instruction")
            print(f"  📊 Max usage: {metrics['usage_stats']['max_usage']}, Min usage: {metrics['usage_stats']['min_usage']}")
            print()
        
        # Calculate overall usage statistics
        all_usage_counts = []
        for split, metrics in overall_metrics.items():
            all_usage_counts.extend([count for count in self.instruction_usage.values() if count > 0])
        
        if all_usage_counts:
            overall_avg_usage = sum(all_usage_counts) / len(all_usage_counts)
            overall_max_usage = max(all_usage_counts)
            overall_min_usage = min(all_usage_counts)
            
            print(f"📊 OVERALL USAGE STATISTICS:")
            print(f"  Average usage across all splits: {overall_avg_usage:.2f}")
            print(f"  Maximum usage across all splits: {overall_max_usage}")
            print(f"  Minimum usage across all splits: {overall_min_usage}")
            print(f"  Total unique instructions used: {len([inst for inst, count in self.instruction_usage.items() if count > 0])}")
        
        print(f"\n✅ Nonsense replacement completed for all splits!")
        print(f"📁 Generated datasets saved to: {self.output_dir}")


def main():
    """Main function for AVDN dataset generation with nonsense instructions."""
    parser = argparse.ArgumentParser(description="Replace AVDN instructions with nonsense instructions")
    parser.add_argument("--nonsense_file", type=str, required=True,
                       help="Path to JSON file containing nonsense instructions")
    parser.add_argument("--output_dir", type=str, default="./generated_avdn_dataset_nonsense",
                       help="Output directory for generated dataset")
    parser.add_argument("--splits", nargs="+", 
                       default=['train', 'val_seen', 'val_unseen'],
                       help="Dataset splits to process")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                       help="Ratio of dataset to sample (default: 1.0 = 100%)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    print(f"🚀 AVDN Dataset Generation with Nonsense Instructions")
    print(f"Nonsense File: {args.nonsense_file}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Sample Ratio: {args.sample_ratio}")

    # Set random seed
    random.seed(args.seed)

    # Create replacer
    replacer = AVDNNonsenseReplacer(
        nonsense_file=args.nonsense_file,
        output_dir=args.output_dir
    )

    # Process all splits
    print(f"\n🚀 Starting AVDN dataset generation with nonsense instructions...")
    
    replacer.process_all_splits(
        splits=args.splits,
        sample_ratio=args.sample_ratio
    )

    print(f"\n✅ AVDN dataset generation with nonsense instructions complete!")
    print(f"📁 Generated dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 