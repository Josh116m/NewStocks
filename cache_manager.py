"""
Cache Management Utility for Feature Engineering
Provides tools to manage, inspect, and clean feature caches.
"""

import argparse
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
import os
from advanced_feature_engineering import AdvancedFeatureEngineer

def get_cache_info(cache_dir: str = "cache/features"):
    """Get information about cached features."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory {cache_dir} does not exist")
        return
    
    cache_files = list(cache_path.glob("*.pkl"))
    
    if not cache_files:
        print(f"📁 Cache directory {cache_dir} is empty")
        return
    
    print(f"📁 Cache Directory: {cache_dir}")
    print(f"📄 Total cache files: {len(cache_files)}")
    print("=" * 80)
    
    total_size = 0
    cache_info = []
    
    for file in cache_files:
        try:
            # Get file info
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            # Load cache data to get metadata
            with open(file, 'rb') as f:
                cache_data = pickle.load(f)
            
            timestamp = cache_data.get('timestamp', 'Unknown')
            feature_type = cache_data.get('feature_type', 'Unknown')
            data_shape = cache_data.get('data', pd.DataFrame()).shape
            
            cache_info.append({
                'file': file.name,
                'size_mb': size_mb,
                'timestamp': timestamp,
                'feature_type': feature_type,
                'data_shape': data_shape
            })
            
        except Exception as e:
            print(f"⚠️ Error reading {file.name}: {e}")
    
    # Sort by timestamp (newest first)
    cache_info.sort(key=lambda x: x['timestamp'] if isinstance(x['timestamp'], datetime) else datetime.min, reverse=True)
    
    # Display cache information
    print(f"{'File':<40} {'Type':<20} {'Size (MB)':<10} {'Shape':<15} {'Created'}")
    print("-" * 100)
    
    for info in cache_info:
        timestamp_str = info['timestamp'].strftime('%Y-%m-%d %H:%M') if isinstance(info['timestamp'], datetime) else str(info['timestamp'])
        print(f"{info['file']:<40} {info['feature_type']:<20} {info['size_mb']:<10.1f} {str(info['data_shape']):<15} {timestamp_str}")
    
    print("-" * 100)
    print(f"Total cache size: {total_size:.1f} MB")

def clear_cache(cache_dir: str = "cache/features", feature_type: str = None, confirm: bool = False):
    """Clear cached features."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory {cache_dir} does not exist")
        return
    
    if feature_type:
        pattern = f"features_{feature_type}_*.pkl"
        files = list(cache_path.glob(pattern))
        action = f"Clear {feature_type} cache files"
    else:
        files = list(cache_path.glob("features_*.pkl"))
        action = "Clear ALL cache files"
    
    if not files:
        print(f"📁 No cache files found to clear")
        return
    
    print(f"🗑️ {action}:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name} ({size_mb:.1f} MB)")
    
    if not confirm:
        response = input(f"\nAre you sure you want to delete {len(files)} files? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled")
            return
    
    deleted_count = 0
    total_size = 0
    
    for file in files:
        try:
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Error deleting {file.name}: {e}")
    
    print(f"✅ Deleted {deleted_count} cache files ({total_size:.1f} MB freed)")

def validate_cache(cache_dir: str = "cache/features"):
    """Validate cached features for integrity."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory {cache_dir} does not exist")
        return
    
    cache_files = list(cache_path.glob("*.pkl"))
    
    if not cache_files:
        print(f"📁 No cache files to validate")
        return
    
    print(f"🔍 Validating {len(cache_files)} cache files...")
    print("=" * 80)
    
    valid_count = 0
    invalid_count = 0
    
    for file in cache_files:
        try:
            with open(file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check required fields
            required_fields = ['data', 'timestamp', 'feature_type']
            missing_fields = [field for field in required_fields if field not in cache_data]
            
            if missing_fields:
                print(f"❌ {file.name}: Missing fields: {missing_fields}")
                invalid_count += 1
                continue
            
            # Check data integrity
            data = cache_data['data']
            if not isinstance(data, pd.DataFrame):
                print(f"❌ {file.name}: Data is not a DataFrame")
                invalid_count += 1
                continue
            
            if data.empty:
                print(f"❌ {file.name}: Data is empty")
                invalid_count += 1
                continue
            
            # Check for required columns
            required_cols = ['ticker', 'date']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                print(f"❌ {file.name}: Missing columns: {missing_cols}")
                invalid_count += 1
                continue
            
            print(f"✅ {file.name}: Valid ({data.shape[0]} rows, {data.shape[1]} columns)")
            valid_count += 1
            
        except Exception as e:
            print(f"❌ {file.name}: Error loading - {e}")
            invalid_count += 1
    
    print("=" * 80)
    print(f"✅ Valid files: {valid_count}")
    print(f"❌ Invalid files: {invalid_count}")
    
    if invalid_count > 0:
        response = input(f"\nDelete {invalid_count} invalid cache files? (y/N): ")
        if response.lower() == 'y':
            # Re-scan and delete invalid files
            for file in cache_files:
                try:
                    with open(file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Quick validation
                    if ('data' not in cache_data or 
                        not isinstance(cache_data['data'], pd.DataFrame) or
                        cache_data['data'].empty):
                        file.unlink()
                        print(f"🗑️ Deleted invalid file: {file.name}")
                        
                except Exception:
                    file.unlink()
                    print(f"🗑️ Deleted corrupted file: {file.name}")

def optimize_cache(cache_dir: str = "cache/features"):
    """Optimize cache by removing duplicate or outdated entries."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory {cache_dir} does not exist")
        return
    
    cache_files = list(cache_path.glob("*.pkl"))
    
    if not cache_files:
        print(f"📁 No cache files to optimize")
        return
    
    print(f"🔧 Optimizing {len(cache_files)} cache files...")
    
    # Group files by feature type
    feature_groups = {}
    
    for file in cache_files:
        try:
            with open(file, 'rb') as f:
                cache_data = pickle.load(f)
            
            feature_type = cache_data.get('feature_type', 'unknown')
            timestamp = cache_data.get('timestamp', datetime.min)
            
            if feature_type not in feature_groups:
                feature_groups[feature_type] = []
            
            feature_groups[feature_type].append({
                'file': file,
                'timestamp': timestamp,
                'data': cache_data
            })
            
        except Exception as e:
            print(f"⚠️ Error reading {file.name}: {e}")
    
    # For each feature type, keep only the most recent cache
    files_to_delete = []
    
    for feature_type, files in feature_groups.items():
        if len(files) > 1:
            # Sort by timestamp (newest first)
            files.sort(key=lambda x: x['timestamp'] if isinstance(x['timestamp'], datetime) else datetime.min, reverse=True)
            
            # Mark older files for deletion
            for file_info in files[1:]:  # Keep the first (newest), delete the rest
                files_to_delete.append(file_info['file'])
                print(f"🗑️ Marking for deletion: {file_info['file'].name} (older {feature_type} cache)")
    
    if files_to_delete:
        print(f"\n🗑️ Found {len(files_to_delete)} outdated cache files")
        response = input(f"Delete {len(files_to_delete)} outdated files? (y/N): ")
        
        if response.lower() == 'y':
            deleted_size = 0
            for file in files_to_delete:
                try:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    deleted_size += size_mb
                    file.unlink()
                    print(f"✅ Deleted: {file.name}")
                except Exception as e:
                    print(f"⚠️ Error deleting {file.name}: {e}")
            
            print(f"✅ Optimization complete! Freed {deleted_size:.1f} MB")
        else:
            print("❌ Optimization cancelled")
    else:
        print("✅ Cache is already optimized")

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Cache Manager")
    parser.add_argument('action', choices=['info', 'clear', 'validate', 'optimize'], 
                       help='Action to perform')
    parser.add_argument('--cache-dir', default='cache/features', 
                       help='Cache directory path (default: cache/features)')
    parser.add_argument('--feature-type', 
                       help='Specific feature type to target (for clear action)')
    parser.add_argument('--confirm', action='store_true', 
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    print("🗂️ Feature Engineering Cache Manager")
    print("=" * 50)
    
    if args.action == 'info':
        get_cache_info(args.cache_dir)
    elif args.action == 'clear':
        clear_cache(args.cache_dir, args.feature_type, args.confirm)
    elif args.action == 'validate':
        validate_cache(args.cache_dir)
    elif args.action == 'optimize':
        optimize_cache(args.cache_dir)

if __name__ == "__main__":
    main()
