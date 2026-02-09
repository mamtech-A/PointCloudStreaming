import os
import sys
import csv
import xml.etree.ElementTree as ET

# Ensure tcp_protocol is importable when run via run.py
# In notebooks, use the notebook's directory instead of __file__
try:
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # When running in Jupyter/notebook environment
    notebook_dir = os.getcwd()
    if 'src' not in os.listdir(notebook_dir):
        # We're likely already in src/
        pass
    else:
        notebook_dir = os.path.join(notebook_dir, 'src')

sys.path.insert(0, notebook_dir)
from tcp_protocol import TCPConnection


# --- Bandwidth Trace Loader ---
def load_bandwidth_trace(log_path):
    """Reads a .log file and returns a list of bandwidth values (bps) for each interval."""
    bandwidths = []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            bytes_received = int(parts[4])
            ms_interval = int(parts[5])
            if ms_interval == 0:
                continue
            bps = (bytes_received * 8) / (ms_interval / 1000)
            bandwidths.append(bps)
    return bandwidths

def parse_mpd_xml(xml_path):
    """Parse MPD XML file (supports both old format and new G-PCC format)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = []
    for frame_elem in root.findall('Frame'):
        frame_id = int(frame_elem.attrib['id'])
        reps = []
        for rep_elem in frame_elem.find('AdaptationSet').findall('Representation'):
            rep = {
                'id': int(rep_elem.attrib['id']),
                'density': int(rep_elem.attrib.get('density', 100)),
                'size': rep_elem.attrib['size'],
                'base_url': rep_elem.find('BaseURL').text,
                'bandwidth': int(rep_elem.attrib.get('bandwidth', 0)),
                'quality': rep_elem.attrib.get('quality', 'unknown'),
                'geom_qp': int(rep_elem.attrib.get('geomQP', 0)),
                'attr_qp': int(rep_elem.attrib.get('attrQP', 0)),
            }
            reps.append(rep)
        frames.append({'id': frame_id, 'representations': reps})
    return frames


def size_to_bytes(size_str):
    """Convert size string (e.g., '1.05M', '419.3K') to bytes"""
    if size_str.endswith('M'):
        return float(size_str[:-1]) * 1024 * 1024
    if size_str.endswith('K'):
        return float(size_str[:-1]) * 1024
    return float(size_str)


def size_to_bits(size_str):
    """Convert size string to bits"""
    return size_to_bytes(size_str) * 8


class PointCloud:
    def __init__(self, points, attributes=None, meta=None):
        self.points = points
        self.attributes = attributes or {}
        self.meta = meta or {}


class DASHPCManifest:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_representations(self, frame_id):
        for frame in self.frames:
            if frame['id'] == frame_id:
                return frame['representations']
        return []



# --- Server Class ---
class PointCloudServer:
    def __init__(self, base_url):
        self.base_url = base_url
        self.pointclouds = {}
        self.manifest = DASHPCManifest()
    def add_pointcloud(self, frame_id, representation_id, pointcloud):
        if frame_id not in self.pointclouds:
            self.pointclouds[frame_id] = {}
        self.pointclouds[frame_id][representation_id] = pointcloud
    def get_manifest(self):
        return self.manifest
    def serve_pointcloud(self, frame_id, representation_id):
        return self.pointclouds[frame_id][representation_id]



# --- Edge Node Class ---
class EdgeNode:
    def __init__(self, server, bandwidth_limit_bps=None, tcp_params=None, abr_algorithm='bandwidth'):
        self.server = server
        self.bandwidth_limit = bandwidth_limit_bps
        self.tcp_params = tcp_params or {}
        self.abr_algorithm = abr_algorithm
        self.last_rep_id = None
        self.last_buffer_level = 0
        self.last_bandwidth = 0
        self.quality_history = []
        self.bandwidth_history = []

    def select_representation(self, frame_id, current_bandwidth_bps=None, buffer_level_s=None):
        manifest = self.server.get_manifest()
        reps = manifest.get_representations(frame_id)
        if not reps:
            return None

        bandwidth = current_bandwidth_bps or self.bandwidth_limit or 1_000_000_000
        self.last_bandwidth = bandwidth
        self.last_buffer_level = buffer_level_s if buffer_level_s is not None else 0
        self.bandwidth_history.append(bandwidth)

        rep_id = self._bandwidth_selection(reps, bandwidth)
        self.last_rep_id = rep_id
        self.quality_history.append(rep_id)
        return rep_id

    def _bandwidth_selection(self, reps, bandwidth_bps):
        reps_sorted = sorted(reps, key=lambda r: r.get('bandwidth', size_to_bits(r['size'])), reverse=True)
        for rep in reps_sorted:
            rep_bandwidth = rep.get('bandwidth', size_to_bits(rep['size']))
            if rep_bandwidth <= bandwidth_bps * 0.8:
                return rep['id']
        return reps_sorted[-1]['id']

    def serve_to_user(self, frame_id, user_id="User", capacity_bps=None, buffer_level_s=None):
        rep_id = self.select_representation(frame_id, current_bandwidth_bps=capacity_bps, buffer_level_s=buffer_level_s)
        pc = self.server.serve_pointcloud(frame_id, rep_id)
        reps = self.server.get_manifest().get_representations(frame_id)
        rep = next((r for r in reps if r['id'] == rep_id), None)
        
        if rep and rep.get('bandwidth'):
            data_bits = rep['bandwidth'] / 30  # bandwidth is per second, we need per frame
        else:
            data_bits = size_to_bits(rep['size']) if rep else 0
        
        tcp = TCPConnection(src="EdgeNode", dst=user_id, **self.tcp_params)
        tcp.establish()
        metrics = tcp.send(int(data_bits / 8) if data_bits else 0, capacity_bps=capacity_bps)
        metrics['packet_log'] = tcp.get_packet_log()
        metrics['rep_info'] = rep
        tcp.close()
        return frame_id, rep_id, pc, metrics





# --- Client Buffer Class ---
class ClientBuffer:
    """
    بافر کلاینت برای مدیریت فریم‌های دریافتی
    شبیه‌سازی واقعی پخش ویدیو با:
    - مصرف همزمان بافر در حین پخش
    - Stall + Rebuffering: وقتی بافر خالی میشه، پخش متوقف و منتظر می‌مونه تا بافر به min_buffer برسه
    - Buffer health monitoring
    - Playback timing
    """
    def __init__(self, target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0):
        """
        target_fps: نرخ پخش هدف (فریم بر ثانیه)
        buffer_capacity_s: حداکثر ظرفیت بافر (ثانیه)
        min_buffer_s: حداقل بافر قبل از شروع پخش (ثانیه)
        """
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps  # مدت زمان هر فریم
        self.buffer_capacity_s = buffer_capacity_s
        self.min_buffer_s = min_buffer_s
        
        # Buffer state
        self.buffer_level_s = 0.0  # سطح فعلی بافر (ثانیه)
        self.frames_in_buffer = []  # لیست فریم‌های در بافر
        
        # Timing
        self.last_update_time = 0.0  # آخرین زمان به‌روزرسانی
        self.playback_started = False  # آیا اولین بار پخش شروع شده
        self.is_playing = False  # آیا الان در حال پخش هستیم (یا در حال rebuffering)
        self.playback_start_time = 0.0
        
        # Rebuffering state
        self.is_rebuffering = False  # آیا در حال rebuffering هستیم
        self.rebuffer_start_time = 0.0  # زمان شروع rebuffering
        
        # Statistics
        self.stall_events = []  # رویدادهای توقف پخش
        self.total_stall_time = 0.0
        self.frames_played = 0
        self.frames_dropped = 0
        self.total_frames_received = 0
        self.buffer_history = []  # تاریخچه سطح بافر
        self.rebuffer_count = 0  # تعداد rebuffering events
        
    def _consume_buffer(self, elapsed_time_s):
        """
        مصرف بافر بر اساس زمان سپری شده (پخش فریم‌ها)
        فقط وقتی is_playing=True مصرف انجام میشه
        Returns: مقدار واقعی پخش شده
        """
        if not self.is_playing:
            return 0.0
        
        # چقدر بافر باید مصرف شود (بر حسب ثانیه)
        consumption_needed = elapsed_time_s
        
        if self.buffer_level_s >= consumption_needed:
            # بافر کافی داریم - مصرف عادی
            self.buffer_level_s -= consumption_needed
            
            # تعداد فریم‌هایی که باید پخش بشن = زمان مصرف شده × FPS
            # ولی نباید بیشتر از فریم‌های موجود در بافر باشه
            frames_to_consume = int(consumption_needed * self.target_fps)
            frames_consumed = min(frames_to_consume, len(self.frames_in_buffer))
            self.frames_played += frames_consumed
            
            # حذف فریم‌ها از لیست
            for _ in range(frames_consumed):
                if self.frames_in_buffer:
                    self.frames_in_buffer.pop(0)
            return consumption_needed
        else:
            # بافر کافی نیست - مصرف تا حد ممکن و سپس توقف پخش
            if self.buffer_level_s > 0:
                consumed = self.buffer_level_s
                
                # پخش همه فریم‌های موجود در بافر
                frames_consumed = len(self.frames_in_buffer)
                self.frames_played += frames_consumed
                
                # خالی کردن بافر
                self.frames_in_buffer.clear()
                self.buffer_level_s = 0.0
                return consumed
            return 0.0
        
    def add_frame(self, frame_id, rep_id, size_bytes, download_time_s, arrival_time_s):
        """
        اضافه کردن فریم به بافر پس از دانلود
        همزمان بافر را بر اساس زمان سپری شده مصرف می‌کند
        
        منطق Rebuffering:
        1. وقتی بافر خالی میشه → پخش متوقف + شروع rebuffering
        2. در حین rebuffering → فقط به بافر اضافه میشه، مصرفی نیست
        3. وقتی بافر به min_buffer رسید → پخش از سر گرفته میشه
        
        Returns: dict با وضعیت بافر
        """
        self.total_frames_received += 1
        
        # محاسبه زمان سپری شده از آخرین به‌روزرسانی
        elapsed_time = arrival_time_s - self.last_update_time if self.last_update_time > 0 else 0
        
        stall_time = 0.0
        event_type = 'buffered'
        
        # اگر در حال پخش هستیم، بافر مصرف کن
        if self.is_playing:
            consumed = self._consume_buffer(elapsed_time)
            
            # آیا بافر خالی شد؟
            if self.buffer_level_s <= 0:
                # شروع Rebuffering!
                self.is_playing = False
                self.is_rebuffering = True
                self.rebuffer_start_time = arrival_time_s - (elapsed_time - consumed) if consumed < elapsed_time else arrival_time_s
                self.rebuffer_count += 1
                # زمان stall = زمان باقیمانده که نتونستیم پخش کنیم
                stall_time = elapsed_time - consumed
                event_type = 'rebuffering_start'
                
        elif self.is_rebuffering:
            # در حال rebuffering - فقط بافر پر میشه، stall ادامه داره
            # stall time = کل زمان سپری شده در این بازه
            stall_time = elapsed_time
        
        # به‌روزرسانی زمان
        self.last_update_time = arrival_time_s
        
        # بررسی overflow (اگر بافر پر باشد، فریم drop شود)
        if self.buffer_level_s >= self.buffer_capacity_s:
            self.frames_dropped += 1
            self.buffer_history.append({
                'time_s': arrival_time_s,
                'buffer_level_s': self.buffer_level_s,
                'event': 'dropped',
                'frame_id': frame_id,
                'stall_time_s': stall_time
            })
            # ثبت stall اگر وجود داشت
            if stall_time > 0:
                self.stall_events.append({
                    'time_s': arrival_time_s,
                    'duration_s': stall_time,
                    'buffer_level_s': self.buffer_level_s,
                    'frame_id': frame_id,
                    'type': 'during_rebuffer'
                })
                self.total_stall_time += stall_time
            return {
                'status': 'dropped',
                'reason': 'buffer_overflow',
                'buffer_level_s': self.buffer_level_s,
                'frame_id': frame_id,
                'stall_time_s': stall_time
            }
        
        # اضافه کردن فریم به بافر
        frame_info = {
            'frame_id': frame_id,
            'rep_id': rep_id,
            'size_bytes': size_bytes,
            'download_time_s': download_time_s,
            'arrival_time_s': arrival_time_s
        }
        self.frames_in_buffer.append(frame_info)
        self.buffer_level_s += self.frame_duration
        
        # محدود کردن به حداکثر ظرفیت
        self.buffer_level_s = min(self.buffer_level_s, self.buffer_capacity_s)
        
        # بررسی شروع پخش (اولین بار)
        if not self.playback_started and self.buffer_level_s >= self.min_buffer_s:
            self.playback_started = True
            self.is_playing = True
            self.playback_start_time = arrival_time_s
            event_type = 'playback_started'
            self.buffer_history.append({
                'time_s': arrival_time_s,
                'buffer_level_s': self.buffer_level_s,
                'event': 'playback_started',
                'frame_id': frame_id,
                'stall_time_s': 0
            })
        
        # بررسی پایان Rebuffering (بافر به min_buffer رسید)
        elif self.is_rebuffering and self.buffer_level_s >= self.min_buffer_s:
            # پایان Rebuffering - پخش از سر گرفته میشه
            rebuffer_duration = arrival_time_s - self.rebuffer_start_time
            self.is_rebuffering = False
            self.is_playing = True
            event_type = 'playback_resumed'
            
            # ثبت کل زمان rebuffering به عنوان یک stall event
            self.stall_events.append({
                'time_s': arrival_time_s,
                'duration_s': rebuffer_duration,
                'buffer_level_s': self.buffer_level_s,
                'frame_id': frame_id,
                'type': 'rebuffer_complete'
            })
            self.total_stall_time += stall_time  # فقط stall این بازه رو اضافه کن
            
            self.buffer_history.append({
                'time_s': arrival_time_s,
                'buffer_level_s': self.buffer_level_s,
                'event': 'playback_resumed',
                'frame_id': frame_id,
                'stall_time_s': rebuffer_duration
            })
            
            return {
                'status': 'buffered',
                'event': 'playback_resumed',
                'buffer_level_s': self.buffer_level_s,
                'buffer_frames': len(self.frames_in_buffer),
                'playback_started': self.playback_started,
                'is_playing': self.is_playing,
                'frame_id': frame_id,
                'stall_time_s': rebuffer_duration,
                'rebuffer_duration_s': rebuffer_duration
            }
        
        # ثبت stall اگر در حال rebuffering هستیم
        if stall_time > 0 and self.is_rebuffering:
            self.total_stall_time += stall_time
        
        # ثبت در تاریخچه
        if event_type == 'buffered':
            self.buffer_history.append({
                'time_s': arrival_time_s,
                'buffer_level_s': self.buffer_level_s,
                'event': 'rebuffering' if self.is_rebuffering else 'buffered',
                'frame_id': frame_id,
                'stall_time_s': stall_time
            })
        
        return {
            'status': 'buffered',
            'event': event_type,
            'buffer_level_s': self.buffer_level_s,
            'buffer_frames': len(self.frames_in_buffer),
            'playback_started': self.playback_started,
            'is_playing': self.is_playing,
            'is_rebuffering': self.is_rebuffering,
            'frame_id': frame_id,
            'stall_time_s': stall_time
        }
    
    def get_stall_info(self):
        """
        اطلاعات stall فعلی
        """
        return {
            'is_rebuffering': self.is_rebuffering,
            'is_playing': self.is_playing,
            'stall_duration_s': self.stall_events[-1]['duration_s'] if self.stall_events else 0,
            'total_stall_time_s': self.total_stall_time,
            'rebuffer_count': self.rebuffer_count
        }
    
    def get_buffer_health(self):
        """
        وضعیت سلامت بافر
        Returns: 'critical', 'low', 'normal', 'high'
        """
        if self.buffer_capacity_s == 0:
            return 'critical'
        ratio = self.buffer_level_s / self.buffer_capacity_s
        if ratio < 0.1:
            return 'critical'
        elif ratio < 0.3:
            return 'low'
        elif ratio < 0.8:
            return 'normal'
        else:
            return 'high'
    
    def get_statistics(self):
        """
        آمار کامل بافر
        """
        return {
            'buffer_level_s': self.buffer_level_s,
            'buffer_level_frames': len(self.frames_in_buffer),
            'buffer_capacity_s': self.buffer_capacity_s,
            'buffer_health': self.get_buffer_health(),
            'playback_started': self.playback_started,
            'is_playing': self.is_playing,
            'is_rebuffering': self.is_rebuffering,
            'frames_played': self.frames_played,
            'frames_received': self.total_frames_received,
            'frames_dropped': self.frames_dropped,
            'stall_count': len(self.stall_events),
            'rebuffer_count': self.rebuffer_count,
            'total_stall_time_s': self.total_stall_time,
            'buffer_utilization': self.buffer_level_s / self.buffer_capacity_s if self.buffer_capacity_s > 0 else 0
        }


# --- Client Class ---
class PointCloudClient:
    def __init__(self, edge_node, target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0):
        self.edge_node = edge_node
        self.current_frame = None
        self.current_representation = None
        
        # Initialize buffer
        self.buffer = ClientBuffer(
            target_fps=target_fps,
            buffer_capacity_s=buffer_capacity_s,
            min_buffer_s=min_buffer_s
        )
        
    def receive_frame(self, frame_id, rep_id, size_bytes, download_time_s, arrival_time_s):
        """
        دریافت فریم از EdgeNode و اضافه کردن به بافر
        """
        return self.buffer.add_frame(frame_id, rep_id, size_bytes, download_time_s, arrival_time_s)
    
    def get_buffer_stats(self):
        return self.buffer.get_statistics()
    
    def get_stall_info(self):
        return self.buffer.get_stall_info()



# --- Simulator Class ---
class Simulator:
    def __init__(self, server, edge_node, clients, duration=10.0):
        self.server = server
        self.edge_node = edge_node
        self.clients = clients
        self.duration = duration
        self.time = 0.0
        
    def run(self, mpd_path, bandwidth_log_path=None):
        print("--- DASH-PC Point Cloud Streaming Simulation ---")
        if not mpd_path or not mpd_path.endswith('.xml') or not os.path.exists(mpd_path):
            raise FileNotFoundError("mpd.xml not found or invalid path.")
        frames = parse_mpd_xml(mpd_path)
        self.server.manifest.frames = frames
        for frame in frames:
            frame_id = frame['id']
            for rep in frame['representations']:
                self.server.add_pointcloud(frame_id, rep['id'], PointCloud(points=None, meta={'density': rep['density']}))
        total_frames = len(frames)
        # Load bandwidth trace if provided
        bandwidths = None
        if bandwidth_log_path:
            bandwidths = load_bandwidth_trace(bandwidth_log_path)
        total_time = 0.0
        
        # تعیین مسیر logs directory
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            # In notebook environment
            project_root = os.path.abspath(os.path.join(os.getcwd(), '..')) if 'src' in os.getcwd() else os.getcwd()
        
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        csv_path = os.path.join(logs_dir, 'results.csv')
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame_id', 'rep_id', 'density', 'size', 'bandwidth_mbps', 'send_time_s',
            'retransmissions', 'cwnd_start', 'cwnd_end', 'srtt_s', 'rto_s',
            'buffer_level_s', 'buffer_health', 'stall', 'stall_duration_s', 'cumulative_time_s'
        ])
        
        # Prepare packet log file
        packet_log_path = os.path.join(logs_dir, 'packets.log')
        packet_log_file = open(packet_log_path, 'w', encoding='utf-8')
        packet_log_file.write('frame_id,time_s,round,packet_num,size_bytes,event,src,dst,cwnd,seq_num,ack_num,rtt_ms\n')
        
        # Prepare buffer log file
        buffer_log_path = os.path.join(logs_dir, 'buffer.log')
        buffer_log_file = open(buffer_log_path, 'w', encoding='utf-8')
        buffer_log_file.write('time_s,frame_id,status,buffer_level_s,buffer_health,stall_duration_s,event_type\n')
        
        for client in self.clients:
            print(f"\n[Client] (User) - Buffer: {client.buffer.buffer_capacity_s}s capacity, {client.buffer.min_buffer_s}s min, {client.buffer.target_fps} FPS target")
            print(f"{'='*100}")
            
            cumulative_time = 0.0  # زمان تجمعی از شروع دانلود
            
            for idx, frame in enumerate(frames):
                frame_id = frame['id']
                # Sliding bandwidth: use bandwidths[idx] if available, else last value
                if bandwidths and len(bandwidths) > 0:
                    if idx < len(bandwidths):
                        bandwidth = bandwidths[idx]
                    else:
                        bandwidth = bandwidths[-1]
                else:
                    bandwidth = self.edge_node.bandwidth_limit or 1_000_000_000
                
                # Get current buffer level for ABR decision
                current_buffer_level = client.buffer.buffer_level_s
                    
                # Let EdgeNode perform TCP send simulation and return metrics
                # Pass buffer level for ABR algorithms that use it
                frame_id, rep_id, pc, metrics = self.edge_node.serve_to_user(
                    frame_id, 
                    user_id="User", 
                    capacity_bps=bandwidth,
                    buffer_level_s=current_buffer_level
                )
                frame_time = metrics.get('time_s', 0)
                total_time += frame_time
                cumulative_time += frame_time
                
                # Get representation info
                rep = next(r for r in frame['representations'] if r['id'] == rep_id)
                
                # Calculate frame size in bytes
                def size_to_bytes(size_str):
                    if size_str.endswith('M'):
                        return float(size_str[:-1]) * 1024 * 1024
                    elif size_str.endswith('K'):
                        return float(size_str[:-1]) * 1024
                    else:
                        return float(size_str)
                frame_size_bytes = size_to_bytes(rep['size'])
                
                # Add frame to client buffer
                buffer_result = client.receive_frame(
                    frame_id=frame_id,
                    rep_id=rep_id,
                    size_bytes=frame_size_bytes,
                    download_time_s=frame_time,
                    arrival_time_s=cumulative_time
                )
                
                # Get stall info from buffer result
                stall_time = buffer_result.get('stall_time_s', 0)
                is_rebuffering = buffer_result.get('is_rebuffering', False)
                is_playing = buffer_result.get('is_playing', False)
                event_type = buffer_result.get('event', 'buffered')
                
                # Get buffer stats
                buffer_stats = client.get_buffer_stats()
                buffer_health = buffer_stats['buffer_health']
                buffer_level = buffer_stats['buffer_level_s']
                
                # Print status with buffer info
                if event_type == 'playback_started':
                    state_indicator = " ▶️ PLAYBACK STARTED"
                elif event_type == 'playback_resumed':
                    state_indicator = f" ▶️ RESUMED (rebuffer={stall_time:.2f}s)"
                elif is_rebuffering:
                    state_indicator = f" ⏸️ REBUFFERING..."
                elif is_playing:
                    state_indicator = " ▶️ PLAYING"
                else:
                    state_indicator = " ⏳ INITIAL BUFFER"
                    
                buffer_emoji = {'critical': '🔴', 'low': '🟡', 'normal': '🟢', 'high': '🔵'}[buffer_health]
                status_icon = "❌" if buffer_result['status'] == 'dropped' else "✅"
                
                print(f"Frame {frame_id:3d}: {status_icon} Rep {rep_id} (density={rep['density']}, size={rep['size']:>5}) | "
                      f"DL={frame_time:.2f}s | BW={bandwidth/1e6:.1f}Mbps | "
                      f"Buffer: {buffer_emoji} {buffer_level:.2f}s ({buffer_health}){state_indicator}")
                
                # Write buffer log
                buffer_log_file.write(f"{cumulative_time:.6f},{frame_id},{buffer_result['status']},{buffer_level:.4f},{buffer_health},{stall_time:.4f},{event_type}\n")
                
                # Write CSV row with buffer info
                csv_writer.writerow([
                    frame_id,
                    rep_id,
                    rep['density'],
                    rep['size'],
                    f"{bandwidth/1e6:.3f}",
                    f"{frame_time:.6f}",
                    metrics.get('retransmissions', 0),
                    metrics.get('cwnd_start', ''),
                    metrics.get('cwnd_end', ''),
                    metrics.get('srtt_s', ''),
                    metrics.get('rto_s', ''),
                    f"{buffer_level:.4f}",
                    buffer_health,
                    'yes' if is_rebuffering else 'no',
                    f"{stall_time:.4f}",
                    f"{cumulative_time:.6f}"
                ])
                
                # Write packet log entries for this frame
                for pkt in metrics.get('packet_log', []):
                    packet_log_file.write(f"{frame_id},{pkt['time_s']:.6f},{pkt['round']},{pkt['packet_num']},{pkt['size_bytes']},{pkt['event']},{pkt['src']},{pkt['dst']},{pkt['cwnd']},{pkt.get('seq_num',0)},{pkt.get('ack_num',0)},{pkt.get('rtt_ms',0):.2f}\n")
            
            # Final statistics
            final_stats = client.get_buffer_stats()
            fps_real = total_frames / total_time if total_time > 0 else 0
            
            print(f"\n{'='*100}")
            print(f"--- Simulation Finished ---")
            print(f"\n📊 Playback Statistics:")
            print(f"   Total Frames: {total_frames}")
            print(f"   Total Download Time: {total_time:.2f}s")
            print(f"   Real Video FPS: {fps_real:.2f} frames per second")
            print(f"\n📦 Buffer Statistics:")
            print(f"   Final Buffer Level: {final_stats['buffer_level_s']:.2f}s ({final_stats['buffer_level_frames']} frames)")
            print(f"   Buffer Health: {final_stats['buffer_health']}")
            print(f"   Frames Played: {final_stats['frames_played']}")
            print(f"   Frames Dropped: {final_stats['frames_dropped']}")
            print(f"\n⚠️  Stall Statistics:")
            print(f"   Rebuffer Events: {final_stats['rebuffer_count']}")
            print(f"   Total Stall Time: {final_stats['total_stall_time_s']:.2f}s")
            if final_stats['rebuffer_count'] > 0:
                print(f"   Average Rebuffer Duration: {final_stats['total_stall_time_s']/final_stats['rebuffer_count']:.3f}s")
            
            # QoE Score (simple formula)
            qoe_score = max(0, 100 - (final_stats['rebuffer_count'] * 10) - (final_stats['total_stall_time_s'] * 5) - (final_stats['frames_dropped'] * 2))
            print(f"\n🎯 QoE Score: {qoe_score:.1f}/100")
            
            csv_file.close()
            packet_log_file.close()
            buffer_log_file.close()
            
            print(f"\n📁 Output Files:")
            print(f"   Per-frame results: {csv_path}")
            print(f"   Packet log: {packet_log_path}")
            print(f"   Buffer log: {buffer_log_path}")
