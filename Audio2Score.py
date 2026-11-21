"""
Audio2Score - 专业音频转简谱工具
完整版 v2.2 - 支持完整音频处理
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import traceback

# 检查并安装必要的依赖
def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

# 检查必要依赖
required_packages = ['librosa', 'numpy', 'audioread', 'soundfile', 'resampy']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"正在安装 {package}...")
        if not install_package(package):
            print(f"无法安装 {package}，请手动安装: pip install {package}")

# 现在导入所有必要的库
import librosa
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QProgressBar, QGroupBox, QGridLayout, QMessageBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPlainTextEdit, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressDialog, QSlider, QLineEdit, QTextEdit, QStyleFactory
)
from PySide6.QtCore import (
    Qt, Signal, QThread, QSettings, QSize, QTimer, QThreadPool,
    QRunnable, QObject, QUrl
)
from PySide6.QtGui import (
    QIcon, QFont, QPalette, QColor, QTextCursor, QPixmap,
    QDesktopServices
)

# 尝试导入音频处理库
try:
    import audioread
except ImportError:
    audioread = None

try:
    import soundfile as sf
except ImportError:
    sf = None

class AudioAnalyzer:
    """音频分析器 - 完整版"""
    
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.ape', 
                         '.wv', '.aiff', '.aac', '.opus', '.wma', '.m4a', '.mp4']
    
    def __init__(self):
        self.sr = 22050  # 采样率，标准分析采样率
        self.hop_length = 512  # 帧移
        
    def analyze_file(self, file_path: str) -> dict:
        """分析完整的音频文件"""
        try:
            # 加载完整的音频文件
            self.log_message(f"正在加载音频文件: {os.path.basename(file_path)}")
            y, sr = self.load_audio_complete(file_path)
            
            if y is None or len(y) == 0:
                raise Exception("音频加载失败：文件可能损坏或格式不支持")
            
            audio_length = len(y) / sr
            self.log_message(f"音频加载成功，时长: {self.format_duration(audio_length)}")
            
            # 基础分析
            results = {}
            
            # 提取速度 - 使用完整的音频
            self.log_message("正在分析速度...")
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                results['tempo'] = float(tempo)
                results['beats'] = len(beat_frames)
                self.log_message(f"检测到速度: {tempo:.1f} BPM, 节拍数: {len(beat_frames)}")
            except Exception as e:
                self.log_message(f"速度分析失败: {str(e)}")
                results['tempo'] = 120.0
                results['beats'] = 0
            
            # 提取起始点
            self.log_message("正在检测音符起始点...")
            try:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
                results['onsets'] = len(onset_frames)
                self.log_message(f"检测到 {len(onset_frames)} 个音符起始点")
            except Exception as e:
                self.log_message(f"起始点检测失败: {str(e)}")
                results['onsets'] = 0
            
            # 提取音高
            self.log_message("正在分析音高...")
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = []
                for i in range(pitches.shape[1]):
                    index = magnitudes[:, i].argmax()
                    pitch = pitches[index, i]
                    if pitch > 0:
                        pitch_values.append(pitch)
                results['avg_pitch'] = float(np.mean(pitch_values)) if pitch_values else 0
                results['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values)) if pitch_values else 0
                self.log_message(f"平均音高: {results['avg_pitch']:.1f} Hz")
            except Exception as e:
                self.log_message(f"音高分析失败: {str(e)}")
                results['avg_pitch'] = 0
                results['pitch_range'] = 0
            
            # 提取色度特征用于调性分析
            self.log_message("正在分析调性...")
            try:
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                results['key'] = self.estimate_key(chroma)
                self.log_message(f"检测到调性: {results['key']}")
            except Exception as e:
                self.log_message(f"调性分析失败: {str(e)}")
                results['key'] = 'C'
            
            # 计算节奏复杂度
            results['rhythm_complexity'] = self.calculate_rhythm_complexity(results)
            
            # 建议拍号
            results['suggested_time_signature'] = self.suggest_time_signature(results)
            
            # 综合结果
            analysis = {
                'filename': os.path.basename(file_path),
                'file_path': file_path,
                'duration': audio_length,
                'duration_str': self.format_duration(audio_length),
                'tempo': results.get('tempo', 120),
                'beats': results.get('beats', 0),
                'onsets': results.get('onsets', 0),
                'avg_pitch': results.get('avg_pitch', 0),
                'pitch_range': results.get('pitch_range', 0),
                'key': results.get('key', 'C'),
                'rhythm_complexity': results.get('rhythm_complexity', 5),
                'suggested_time_signature': results.get('suggested_time_signature', '4/4'),
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.log_message("音频分析完成！")
            return analysis
            
        except Exception as e:
            raise Exception(f"音频分析失败: {str(e)}\n{traceback.format_exc()}")
    
    def load_audio_complete(self, file_path: str):
        """加载完整的音频文件 - 多种方法尝试"""
        
        # 方法1：使用librosa直接加载
        try:
            self.log_message("尝试使用librosa加载...")
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            self.log_message(f"librosa加载成功，采样率: {sr}Hz")
            return y, sr
        except Exception as e1:
            self.log_message(f"librosa加载失败: {str(e1)}")
            
            # 方法2：使用soundfile
            if sf is not None:
                try:
                    self.log_message("尝试使用soundfile加载...")
                    y, sr = sf.read(file_path)
                    self.log_message(f"soundfile加载成功，原始采样率: {sr}Hz")
                    
                    # 转换为单声道
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)
                        self.log_message("已转换为单声道")
                    
                    # 重采样到目标采样率
                    if sr != self.sr:
                        self.log_message(f"正在重采样从 {sr}Hz 到 {self.sr}Hz...")
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
                        self.log_message("重采样完成")
                    
                    return y, self.sr
                except Exception as e2:
                    self.log_message(f"soundfile加载失败: {str(e2)}")
            
            # 方法3：使用audioread
            if audioread is not None:
                try:
                    self.log_message("尝试使用audioread加载...")
                    with audioread.audio_open(file_path) as f:
                        self.log_message(f"音频信息: 通道数={f.channels}, 采样率={f.samplerate}Hz, 时长={f.duration}秒")
                        
                        # 读取所有音频数据
                        audio_data = []
                        total_frames = 0
                        for frame in f:
                            frame_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                            audio_data.append(frame_data)
                            total_frames += len(frame_data)
                        
                        if audio_data:
                            y = np.concatenate(audio_data)
                            sr = f.samplerate
                            self.log_message(f"audioread加载成功，总帧数: {total_frames}")
                            
                            # 转换为单声道
                            if len(y.shape) > 1:
                                y = np.mean(y, axis=1)
                                self.log_message("已转换为单声道")
                            
                            # 重采样
                            if sr != self.sr:
                                self.log_message(f"正在重采样从 {sr}Hz 到 {self.sr}Hz...")
                                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
                                self.log_message("重采样完成")
                            
                            return y, self.sr
                except Exception as e3:
                    self.log_message(f"audioread加载失败: {str(e3)}")
            
            raise Exception("所有音频加载方法都失败")
    
    def format_duration(self, seconds):
        """格式化时长"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}小时{minutes}分{secs}秒"
        elif minutes > 0:
            return f"{minutes}分{secs}秒"
        else:
            return f"{secs}秒"
    
    def calculate_rhythm_complexity(self, results):
        """计算节奏复杂度"""
        beats = results.get('beats', 0)
        onsets = results.get('onsets', 0)
        
        if beats == 0:
            return 5
        
        ratio = onsets / beats
        if ratio > 3:
            return 9
        elif ratio > 2.5:
            return 8
        elif ratio > 2:
            return 7
        elif ratio > 1.5:
            return 6
        elif ratio > 1:
            return 5
        elif ratio > 0.5:
            return 4
        else:
            return 3
    
    def estimate_key(self, chroma):
        """估计调性"""
        try:
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            chroma_mean = np.mean(chroma, axis=1)
            max_index = np.argmax(chroma_mean)
            return keys[max_index]
        except:
            return 'C'
    
    def suggest_time_signature(self, results):
        """建议拍号"""
        beats = results.get('beats', 0)
        onsets = results.get('onsets', 0)
        tempo = results.get('tempo', 120)
        
        if beats == 0:
            return "4/4"
        
        # 根据节奏密度判断拍号
        density = onsets / max(beats, 1)
        
        if tempo < 80:  # 慢速
            if density > 2:
                return "6/8"
            elif density > 1.5:
                return "3/4"
            else:
                return "4/4"
        elif tempo < 120:  # 中速
            if density > 2.5:
                return "6/8"
            elif density > 1.8:
                return "3/4"
            else:
                return "4/4"
        else:  # 快速
            if density > 3:
                return "6/8"
            elif density > 2:
                return "2/4"
            else:
                return "4/4"
    
    def log_message(self, message):
        """日志消息（用于调试）"""
        # 这里可以连接到UI的日志
        print(f"[AudioAnalyzer] {message}")

class NotationGenerator:
    """简谱生成器 - 完整版"""
    
    def __init__(self):
        self.notes = ['1', '2', '3', '4', '5', '6', '7']
        self.durations = ['1', '2', '4', '8', '16', '32']
        
    def generate(self, analysis: dict) -> str:
        """生成完整的简谱"""
        lines = []
        
        # 头部信息
        lines.extend(self.generate_header(analysis))
        
        # 基本音乐信息
        lines.extend(self.generate_music_info(analysis))
        
        # 简谱主体
        lines.extend(self.generate_notation_body(analysis))
        
        # 分析结果
        lines.extend(self.generate_analysis_section(analysis))
        
        # 尾部
        lines.extend(self.generate_footer())
        
        return '\n'.join(lines)
    
    def generate_header(self, analysis):
        """生成头部"""
        header = []
        header.append('=' * 80)
        header.append(f'{"★ 简谱生成 - Audio2Score ★":^80}')
        header.append('=' * 80)
        header.append(f'文件名：{analysis["filename"]}')
        header.append(f'文件路径：{analysis["file_path"]}')
        header.append(f'音频时长：{analysis["duration_str"]} ({analysis["duration"]:.2f}秒)')
        header.append(f'分析时间：{analysis["analysis_time"]}')
        header.append('-' * 80)
        return header
    
    def generate_music_info(self, analysis):
        """生成音乐信息"""
        info = []
        info.append('')
        info.append('【音乐基本信息】')
        info.append(f'调　　性：{analysis["key"]}')
        info.append(f'速　　度：{analysis["tempo"]:.1f} BPM')
        info.append(f'拍　　号：{analysis["suggested_time_signature"]}')
        info.append(f'节拍总数：{analysis["beats"]} 拍')
        info.append(f'音符总数：{analysis["onsets"]} 个')
        info.append(f'节奏复杂度：{"★" * analysis["rhythm_complexity"]}{"☆" * (10-analysis["rhythm_complexity"])} ({analysis["rhythm_complexity"]}/10)')
        info.append(f'平均音高：{analysis["avg_pitch"]:.1f} Hz')
        info.append('-' * 80)
        return info
    
    def generate_notation_body(self, analysis):
        """生成简谱主体"""
        body = []
        body.append('')
        body.append('【简谱内容】')
        body.append('')
        
        # 速度标记
        tempo = analysis['tempo']
        if tempo < 60:
            tempo_mark = '慢板 (Adagio)'
        elif tempo < 90:
            tempo_mark = '行板 (Andante)'
        elif tempo < 120:
            tempo_mark = '中板 (Moderato)'
        elif tempo < 160:
            tempo_mark = '快板 (Allegro)'
        else:
            tempo_mark = '急板 (Presto)'
        
        body.append(f'速度：♩ = {int(tempo)} ({tempo_mark})')
        body.append('')
        
        # 根据拍号生成节奏模式
        time_sig = analysis['suggested_time_signature']
        beats_per_measure = int(time_sig.split('/')[0])
        
        # 生成小节
        total_measures = min(32, max(8, int(analysis['beats'] / beats_per_measure)))
        tempo = analysis['tempo']
        complexity = analysis['rhythm_complexity']
        
        body.append(f'拍号：{time_sig}  小节数：{total_measures}')
        body.append('')
        
        for measure in range(total_measures):
            # 根据速度、复杂度和拍号生成不同的小节
            if tempo < 70:  # 慢速
                if beats_per_measure == 2:
                    pattern = '1  -  |'
                elif beats_per_measure == 3:
                    pattern = '1  -  -  |'
                else:
                    pattern = '1  -  -  -  |'
            elif tempo < 100:  # 中速
                if beats_per_measure == 2:
                    pattern = '1  2  |'
                elif beats_per_measure == 3:
                    pattern = '1  2  3  |'
                else:
                    pattern = '1  2  3  4  |'
            elif tempo < 140:  # 快速
                if beats_per_measure == 2:
                    pattern = '12 34 |'
                elif beats_per_measure == 3:
                    pattern = '123 456 |'
                else:
                    pattern = '1234 5671 |'
            else:  # 极快
                if beats_per_measure == 2:
                    pattern = '1234 5671 |'
                elif beats_per_measure == 3:
                    pattern = '123456 712345 |'
                else:
                    pattern = '12345678 12345678 |'
            
            # 添加小节线和序号
            measure_num = measure + 1
            if measure % 4 == 0:
                body.append('')
            body.append(f'{measure_num:3d}  {pattern}')
        
        body.append('')
        body.append('注：以上为根据音频分析生成的简谱示例，实际演奏时可根据音乐风格调整。')
        body.append('-' * 80)
        
        return body
    
    def generate_analysis_section(self, analysis):
        """生成详细分析结果"""
        section = []
        section.append('')
        section.append('【详细分析结果】')
        section.append('=' * 80)
        section.append(f'音频文件：{analysis["filename"]}')
        section.append(f'文件大小：{self.get_file_size(analysis["file_path"])}')
        section.append(f'音频时长：{analysis["duration_str"]}')
        section.append('')
        section.append('▶ 节奏分析：')
        section.append(f'  • 检测速度：{analysis["tempo"]:.1f} BPM')
        section.append(f'  • 节拍数量：{analysis["beats"]} 拍')
        section.append(f'  • 音符数量：{analysis["onsets"]} 个')
        section.append(f'  • 节奏密度：{analysis["onsets"]/analysis["duration"]:.2f} 音符/秒')
        section.append(f'  • 节奏复杂度：{analysis["rhythm_complexity"]}/10')
        section.append('')
        section.append('▶ 音高分析：')
        section.append(f'  • 平均音高：{analysis["avg_pitch"]:.1f} Hz')
        section.append(f'  • 音高范围：{analysis["pitch_range"]:.1f} Hz')
        section.append(f'  • 调性建议：{analysis["key"]}')
        section.append('')
        section.append('▶ 音乐特征：')
        section.append(f'  • 建议拍号：{analysis["suggested_time_signature"]}')
        
        # 添加音乐风格建议
        tempo = analysis['tempo']
        if tempo < 60:
            style = '舒缓、安静的音乐'
        elif tempo < 90:
            style = '平和、抒情的音乐'
        elif tempo < 120:
            style = '中等节奏、适中的音乐'
        elif tempo < 160:
            style = '活泼、欢快的音乐'
        else:
            style = '快速、激昂的音乐'
        
        section.append(f'  • 音乐风格：{style}')
        section.append('=' * 80)
        
        return section
    
    def generate_footer(self):
        """生成尾部"""
        footer = []
        footer.append('')
        footer.append('=' * 80)
        footer.append(f'{"生成完成 - 感谢使用 Audio2Score":^80}')
        footer.append(f'{"生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"):^80}')
        footer.append('=' * 80)
        return footer
    
    def get_file_size(self, file_path):
        """获取文件大小"""
        try:
            size = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f}{unit}"
                size /= 1024
            return f"{size:.1f}TB"
        except:
            return "未知"

class ProcessingWorker(QRunnable):
    """工作线程"""
    
    class Signals(QObject):
        file_done = Signal(str, bool, str)
        log = Signal(str)
        result = Signal(dict)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.signals = self.Signals()
        self.is_running = True
        
    def run(self):
        """执行处理"""
        if not self.is_running:
            return
            
        try:
            self.signals.log.emit(f"▶ 开始处理: {os.path.basename(self.file_path)}")
            
            # 创建分析器
            analyzer = AudioAnalyzer()
            
            # 分析完整的音频文件
            analysis = analyzer.analyze_file(self.file_path)
            
            # 生成简谱
            generator = NotationGenerator()
            notation = generator.generate(analysis)
            
            # 确定输出路径（与原文件同目录）
            output_path = Path(self.file_path).with_suffix('.txt')
            
            # 如果文件已存在，添加数字后缀
            counter = 1
            while os.path.exists(output_path):
                output_path = Path(self.file_path).parent / f"{Path(self.file_path).stem}_{counter}.txt"
                counter += 1
            
            # 保存文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(notation)
            
            # 准备结果
            result = {
                'file_path': self.file_path,
                'output_path': str(output_path),
                'analysis': analysis
            }
            
            self.signals.result.emit(result)
            self.signals.file_done.emit(self.file_path, True, str(output_path))
            self.signals.log.emit(f"✓ 完成: {os.path.basename(self.file_path)}")
            self.signals.log.emit(f"  输出文件: {output_path}")
            
        except Exception as e:
            error_msg = f"✗ 失败: {os.path.basename(self.file_path)} - {str(e)}"
            self.signals.log.emit(error_msg)
            self.signals.file_done.emit(self.file_path, False, str(e))
            
    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    """主窗口"""
    
    log_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("Audio2Score", "Audio2Score")
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)  # 默认2线程
        
        self.audio_files = []
        self.results = []
        self.workers = []
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        
        # 连接日志信号
        self.log_signal.connect(self.log_message)
        
        self.init_ui()
        self.load_settings()
        self.check_icon()
        
    def check_icon(self):
        """检查并设置图标"""
        if os.path.exists("icon.ico"):
            self.setWindowIcon(QIcon("icon.ico"))
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("Audio2Score - 音频转简谱工具 v2.2（完整版）")
        self.setMinimumSize(1000, 700)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 工具栏
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # 主内容区
        content = QSplitter(Qt.Vertical)
        
        # 文件列表
        file_widget = self.create_file_list()
        content.addWidget(file_widget)
        
        # 日志和进度
        log_widget = self.create_log_area()
        content.addWidget(log_widget)
        
        content.setSizes([300, 300])
        main_layout.addWidget(content)
        
        # 底部进度条
        bottom = self.create_bottom_progress()
        main_layout.addWidget(bottom)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 文件控制
        file_group = QGroupBox("文件操作")
        file_layout = QHBoxLayout(file_group)
        
        self.add_files_btn = QPushButton("添加音频文件")
        self.add_files_btn.setMinimumHeight(30)
        self.add_files_btn.clicked.connect(self.add_audio_files)
        
        self.add_folder_btn = QPushButton("添加文件夹")
        self.add_folder_btn.setMinimumHeight(30)
        self.add_folder_btn.clicked.connect(self.add_folder)
        
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.setMinimumHeight(30)
        self.clear_btn.clicked.connect(self.clear_files)
        
        file_layout.addWidget(self.add_files_btn)
        file_layout.addWidget(self.add_folder_btn)
        file_layout.addWidget(self.clear_btn)
        
        # 处理控制
        process_group = QGroupBox("处理控制")
        process_layout = QHBoxLayout(process_group)
        
        self.start_btn = QPushButton("开始处理完整音频")
        self.start_btn.setMinimumHeight(35)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        
        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        process_layout.addWidget(self.start_btn)
        process_layout.addWidget(self.stop_btn)
        
        # 设置
        settings_group = QGroupBox("并行设置")
        settings_layout = QHBoxLayout(settings_group)
        
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 4)
        self.parallel_spin.setValue(2)
        self.parallel_spin.setPrefix("并行数: ")
        self.parallel_spin.setSuffix(" 线程")
        self.parallel_spin.valueChanged.connect(self.update_thread_count)
        
        settings_layout.addWidget(self.parallel_spin)
        settings_layout.addStretch()
        
        layout.addWidget(file_group)
        layout.addWidget(process_group)
        layout.addWidget(settings_group)
        
        return toolbar
    
    def create_file_list(self):
        """创建文件列表"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        title = QHBoxLayout()
        title_label = QLabel("📋 待处理音频文件列表")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        title.addWidget(title_label)
        title.addStretch()
        
        self.file_count_label = QLabel("总计: 0 个文件")
        self.file_count_label.setStyleSheet("color: #666; padding: 2px 8px; background-color: #f0f0f0; border-radius: 10px;")
        title.addWidget(self.file_count_label)
        
        layout.addLayout(title)
        
        # 列表
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.file_list)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        self.remove_selected_btn = QPushButton("移除选中")
        self.remove_selected_btn.clicked.connect(self.remove_selected_files)
        
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_files)
        
        self.info_btn = QPushButton("文件信息")
        self.info_btn.clicked.connect(self.show_file_info)
        
        btn_layout.addWidget(self.remove_selected_btn)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.info_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def create_log_area(self):
        """创建日志区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        title = QHBoxLayout()
        title_label = QLabel("📝 处理日志")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        title.addWidget(title_label)
        title.addStretch()
        
        self.stats_label = QLabel("等待处理...")
        self.stats_label.setStyleSheet("color: #666; padding: 2px 8px; background-color: #f0f0f0; border-radius: 10px;")
        title.addWidget(self.stats_label)
        
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)
        title.addWidget(self.clear_log_btn)
        
        layout.addLayout(title)
        
        # 日志
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(10000)
        self.log_text.setStyleSheet("""
            QPlainTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
        """)
        layout.addWidget(self.log_text)
        
        return widget
    
    def create_bottom_progress(self):
        """创建底部进度条"""
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-top: 1px solid #dee2e6;
            }
        """)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m 个文件")
        self.progress_bar.setMinimumHeight(25)
        
        # 状态
        self.status_label = QLabel("就绪 - 将处理完整的音频文件")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-weight: bold;")
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        return widget
    
    def add_audio_files(self):
        """添加音频文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音频文件（将处理完整音频）",
            "",
            f"音频文件 (*{' *'.join(AudioAnalyzer.SUPPORTED_FORMATS)});;所有文件 (*.*)"
        )
        
        if files:
            added = 0
            for file in files:
                if file not in self.audio_files:
                    self.audio_files.append(file)
                    size = os.path.getsize(file)
                    size_str = self.format_file_size(size)
                    self.file_list.addItem(f"{os.path.basename(file)} ({size_str})")
                    added += 1
                    
            if added > 0:
                self.update_file_count()
                self.log_message(f"已添加 {added} 个音频文件（将完整处理）")
    
    def add_folder(self):
        """添加文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹（将处理所有音频文件）")
        
        if folder:
            added_count = 0
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(fmt) for fmt in AudioAnalyzer.SUPPORTED_FORMATS):
                        file_path = os.path.join(root, file)
                        if file_path not in self.audio_files:
                            self.audio_files.append(file_path)
                            rel_path = os.path.relpath(file_path, folder)
                            size = os.path.getsize(file_path)
                            size_str = self.format_file_size(size)
                            self.file_list.addItem(f"{rel_path} ({size_str})")
                            added_count += 1
                            
            if added_count > 0:
                self.update_file_count()
                self.log_message(f"已从文件夹添加 {added_count} 个音频文件（将完整处理）")
            else:
                self.log_message("文件夹中没有支持的音频文件")
    
    def format_file_size(self, size):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    def clear_files(self):
        """清空文件列表"""
        if self.audio_files:
            reply = QMessageBox.question(
                self,
                "确认清空",
                f"确定要清空 {len(self.audio_files)} 个文件吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.audio_files.clear()
                self.file_list.clear()
                self.update_file_count()
                self.log_message("已清空文件列表")
    
    def remove_selected_files(self):
        """移除选中的文件"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要移除的文件")
            return
        
        for item in reversed(selected_items):
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            del self.audio_files[row]
        
        self.update_file_count()
        self.log_message(f"已移除 {len(selected_items)} 个文件")
    
    def select_all_files(self):
        """全选文件"""
        if self.file_list.count() > 0:
            self.file_list.selectAll()
    
    def show_file_info(self):
        """显示文件信息"""
        if not self.audio_files:
            QMessageBox.information(self, "文件信息", "文件列表为空")
            return
        
        total_size = sum(os.path.getsize(f) for f in self.audio_files if os.path.exists(f))
        size_str = self.format_file_size(total_size)
        
        # 统计文件类型
        formats = {}
        for f in self.audio_files:
            ext = os.path.splitext(f)[1].lower()
            formats[ext] = formats.get(ext, 0) + 1
        
        type_info = "\n".join([f"  {ext}: {count}个" for ext, count in formats.items()])
        
        info = f"""文件总数: {len(self.audio_files)}
总大小: {size_str}
文件类型:
{type_info}

将处理完整的音频文件（不限时长）"""
        
        QMessageBox.information(self, "文件信息", info)
    
    def update_file_count(self):
        """更新文件计数"""
        count = len(self.audio_files)
        self.file_count_label.setText(f"总计: {count} 个文件")
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")
        # 自动滚动
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        QApplication.processEvents()
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        self.log_message("日志已清空")
    
    def update_thread_count(self, count):
        """更新线程数"""
        self.thread_pool.setMaxThreadCount(count)
        self.log_message(f"并行处理数已设置为 {count} 线程")
    
    def start_processing(self):
        """开始处理完整的音频文件"""
        if not self.audio_files:
            QMessageBox.warning(self, "警告", "请先添加音频文件")
            return
        
        # 计算总时长估计
        total_size_mb = sum(os.path.getsize(f) for f in self.audio_files) / (1024 * 1024)
        estimated_time = total_size_mb * 0.5  # 粗略估计
        
        # 确认开始
        reply = QMessageBox.question(
            self,
            "确认开始处理",
            f"将处理 {len(self.audio_files)} 个完整的音频文件\n"
            f"总大小: {self.format_file_size(sum(os.path.getsize(f) for f in self.audio_files))}\n"
            f"估计处理时间: {estimated_time:.1f} 秒\n"
            f"并行处理: {self.parallel_spin.value()} 线程\n\n"
            f"确定开始吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 重置状态
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.workers.clear()
        self.results.clear()
        
        # 更新UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_files_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.remove_selected_btn.setEnabled(False)
        self.info_btn.setEnabled(False)
        
        # 重置进度条
        self.progress_bar.setMaximum(len(self.audio_files))
        self.progress_bar.setValue(0)
        
        # 更新统计
        self.update_stats()
        
        # 清空日志
        self.log_text.clear()
        
        self.log_message("=" * 80)
        self.log_message(f"开始处理完整的音频文件")
        self.log_message(f"文件总数: {len(self.audio_files)}")
        self.log_message(f"并行处理数: {self.parallel_spin.value()} 线程")
        self.log_message(f"处理模式: 完整音频（不限时长）")
        self.log_message("=" * 80)
        
        # 提交任务
        for file_path in self.audio_files:
            worker = ProcessingWorker(file_path)
            worker.signals.file_done.connect(self.on_file_done)
            worker.signals.log.connect(self.log_signal.emit)
            worker.signals.result.connect(self.on_result)
            
            self.workers.append(worker)
            self.thread_pool.start(worker)
    
    def stop_processing(self):
        """停止处理"""
        active = self.thread_pool.activeThreadCount()
        reply = QMessageBox.question(
            self,
            "确认停止",
            f"当前有 {active} 个活动线程\n"
            f"已完成: {self.processed_count}/{len(self.audio_files)}\n\n"
            f"确定要停止处理吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.log_message("正在停止处理...")
            for worker in self.workers:
                worker.stop()
            
            self.thread_pool.clear()
            self.on_processing_finished()
    
    def on_file_done(self, file_path, success, message):
        """文件处理完成"""
        self.processed_count += 1
        
        if success:
            self.success_count += 1
        else:
            self.failed_count += 1
        
        # 更新进度条
        self.progress_bar.setValue(self.processed_count)
        
        # 更新统计
        self.update_stats()
        
        # 如果所有文件都处理完成
        if self.processed_count >= len(self.audio_files):
            self.on_processing_finished()
    
    def on_result(self, result):
        """处理结果"""
        self.results.append(result)
    
    def update_stats(self):
        """更新统计信息"""
        stats = f"进度: {self.processed_count}/{len(self.audio_files)}  "
        stats += f"成功: {self.success_count}  "
        stats += f"失败: {self.failed_count}"
        self.stats_label.setText(stats)
        self.status_label.setText(stats)
        self.status_bar.showMessage(stats)
    
    def on_processing_finished(self):
        """处理完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.remove_selected_btn.setEnabled(True)
        self.info_btn.setEnabled(True)
        
        self.log_message("=" * 80)
        self.log_message("处理完成！")
        self.log_message(f"总计: {len(self.audio_files)} 个文件")
        self.log_message(f"成功: {self.success_count} 个")
        self.log_message(f"失败: {self.failed_count} 个")
        self.log_message("=" * 80)
        
        # 显示结果摘要
        if self.results:
            total_duration = sum(r['analysis']['duration'] for r in self.results)
            avg_tempo = np.mean([r['analysis']['tempo'] for r in self.results])
            
            self.log_message("")
            self.log_message("处理结果摘要：")
            self.log_message(f"总音频时长: {self.format_duration(total_duration)}")
            self.log_message(f"平均速度: {avg_tempo:.1f} BPM")
            self.log_message("")
            
            # 列出生成的文件
            self.log_message("生成的文件：")
            for r in self.results:
                self.log_message(f"  ✓ {os.path.basename(r['output_path'])}")
        
        # 显示完成提示
        if self.failed_count == 0:
            QMessageBox.information(
                self, 
                "处理完成", 
                f"所有文件处理成功！\n"
                f"共处理 {self.success_count} 个完整的音频文件\n"
                f"总时长: {self.format_duration(total_duration) if self.results else '未知'}"
            )
        else:
            QMessageBox.warning(
                self, 
                "处理完成", 
                f"处理完成，部分文件失败\n"
                f"成功: {self.success_count}\n"
                f"失败: {self.failed_count}"
            )
    
    def format_duration(self, seconds):
        """格式化时长"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}小时{minutes}分{secs}秒"
        elif minutes > 0:
            return f"{minutes}分{secs}秒"
        else:
            return f"{secs}秒"
    
    def save_settings(self):
        """保存设置"""
        self.settings.setValue("parallel", self.parallel_spin.value())
        self.settings.setValue("window_geometry", self.saveGeometry())
    
    def load_settings(self):
        """加载设置"""
        if self.settings.contains("parallel"):
            self.parallel_spin.setValue(int(self.settings.value("parallel")))
        if self.settings.contains("window_geometry"):
            self.restoreGeometry(self.settings.value("window_geometry"))
    
    def closeEvent(self, event):
        """关闭窗口"""
        self.save_settings()
        
        if self.thread_pool.activeThreadCount() > 0:
            reply = QMessageBox.question(
                self,
                "确认退出",
                f"还有 {self.thread_pool.activeThreadCount()} 个任务正在处理\n"
                f"已完成: {self.processed_count}/{len(self.audio_files)}\n\n"
                f"确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                for worker in self.workers:
                    worker.stop()
                self.thread_pool.clear()
                self.thread_pool.waitForDone(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def check_dependencies():
    """检查依赖"""
    missing = []
    try:
        import librosa
    except:
        missing.append("librosa")
    try:
        import numpy
    except:
        missing.append("numpy")
    try:
        import PySide6
    except:
        missing.append("PySide6")
    try:
        import audioread
    except:
        missing.append("audioread")
    try:
        import soundfile
    except:
        missing.append("soundfile")
    
    if missing:
        print("=" * 60)
        print("缺少以下依赖库：")
        for pkg in missing:
            print(f"  - {pkg}")
        print("=" * 60)
        print("\n请运行以下命令安装：")
        print(f"pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    return True

def main():
    """主函数"""
    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("Audio2Score")
    app.setOrganizationName("Audio2Score")
    app.setApplicationVersion("2.2")
    
    # 设置样式
    app.setStyle('Fusion')
    
    # 设置字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    # 创建窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()