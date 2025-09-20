#!/usr/bin/env python3
"""
체크포인트 관리자 - 전처리기 저장/로드 및 진행상황 관리
"""

import os
import pickle
import json

class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_preprocessor(self, preprocessor, filename='preprocessor.pkl'):
        """전처리기 저장"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"✅ 전처리기 저장: {filepath}")

    def load_preprocessor(self, filename='preprocessor.pkl'):
        """전처리기 로드"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                preprocessor = pickle.load(f)
            print(f"✅ 전처리기 로드: {filepath}")
            return preprocessor
        return None

    def save_progress(self, progress_info):
        """진행 상황 저장"""
        filepath = os.path.join(self.checkpoint_dir, 'progress.json')
        with open(filepath, 'w') as f:
            json.dump(progress_info, f, indent=2)
        print(f"📊 진행상황 저장: {progress_info['current_chunk']}/{progress_info['total_chunks']}")

    def load_progress(self):
        """진행 상황 로드"""
        filepath = os.path.join(self.checkpoint_dir, 'progress.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                progress = json.load(f)
            print(f"📊 진행상황 로드: {progress['current_chunk']}/{progress['total_chunks']}")
            return progress
        return None

    def list_processed_chunks(self):
        """처리된 청크 파일 목록"""
        chunk_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith('chunk_') and file.endswith('.parquet'):
                chunk_files.append(os.path.join(self.checkpoint_dir, file))
        return sorted(chunk_files)

    def clear_checkpoints(self):
        """체크포인트 파일들 삭제"""
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            print("✅ 체크포인트 파일 삭제 완료")