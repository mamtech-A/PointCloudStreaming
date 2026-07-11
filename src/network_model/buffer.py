"""Client-side playback buffer and client wrapper.

Moved verbatim from the legacy network_model.py (unchanged logic).
"""


class ClientBuffer:
    """
    بافر کلاینت برای مدیریت فریم‌های دریافتی
    شبیه‌سازی واقعی پخش ویدیو با:
    - مصرف همزمان بافر در حین پخش
    - Stall + Rebuffering: وقتی بافر خالی میشه، پخش متوقف و منتظر می‌مونه تا بافر به min_buffer برسه
    - Buffer health monitoring
    - Playback timing
    """
    def __init__(self, target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0,
                 playback_rate_min=1.0):
        """
        target_fps: نرخ پخش هدف (فریم بر ثانیه)
        buffer_capacity_s: حداکثر ظرفیت بافر (ثانیه)
        min_buffer_s: حداقل بافر قبل از شروع پخش (ثانیه)
        playback_rate_min: ADAPTIVE PLAYBACK RATE floor (dash.js-style AMP).
            1.0 (default) = fixed-rate playback, exactly the legacy behavior.
            0.9 = when the buffer dips below min_buffer_s, playback slows toward
            this floor instead of racing into a stall — subjective studies
            (Drop-or-Stop QoMEX'24 / ACM TOMM'26) find <=0.9x imperceptible and
            always preferred over rebuffering.
        """
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps  # مدت زمان هر فریم
        self.buffer_capacity_s = buffer_capacity_s
        self.min_buffer_s = min_buffer_s
        self.playback_rate_min = float(playback_rate_min)
        
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
        # Adaptive-playback statistics (all zero when playback_rate_min == 1.0)
        self.slowdown_time_s = 0.0    # wall seconds played at rate < 1.0
        self.slowdown_integral = 0.0  # integral of (1 - rate) dt over played time

    def _playback_rate(self):
        """Current playback rate: 1.0 with a healthy buffer, ramping down to
        playback_rate_min as the buffer drains below min_buffer_s."""
        if self.playback_rate_min >= 1.0 or self.min_buffer_s <= 0:
            return 1.0
        if self.buffer_level_s >= self.min_buffer_s:
            return 1.0
        return max(self.playback_rate_min, self.buffer_level_s / self.min_buffer_s)
        
    def _consume_buffer(self, elapsed_time_s):
        """
        مصرف بافر بر اساس زمان سپری شده (پخش فریم‌ها)
        فقط وقتی is_playing=True مصرف انجام میشه
        Returns: مقدار واقعی پخش شده
        """
        if not self.is_playing:
            return 0.0

        # Adaptive playback: at rate r, E wall-seconds consume E*r CONTENT-seconds
        # of buffer. r == 1.0 (default) reproduces the legacy fixed-rate behavior
        # exactly. Return value is WALL seconds actually played (callers compute
        # stall = elapsed - returned).
        rate = self._playback_rate()
        # چقدر بافر باید مصرف شود (بر حسب ثانیه‌ی محتوا)
        consumption_needed = elapsed_time_s * rate

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
            if rate < 1.0:
                self.slowdown_time_s += elapsed_time_s
                self.slowdown_integral += (1.0 - rate) * elapsed_time_s
            return elapsed_time_s
        else:
            # بافر کافی نیست - مصرف تا حد ممکن و سپس توقف پخش
            if self.buffer_level_s > 0:
                consumed_content = self.buffer_level_s
                consumed_wall = min(elapsed_time_s, consumed_content / rate)

                # پخش همه فریم‌های موجود در بافر
                frames_consumed = len(self.frames_in_buffer)
                self.frames_played += frames_consumed

                # خالی کردن بافر
                self.frames_in_buffer.clear()
                self.buffer_level_s = 0.0
                if rate < 1.0:
                    self.slowdown_time_s += consumed_wall
                    self.slowdown_integral += (1.0 - rate) * consumed_wall
                return consumed_wall
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
            'playback_rate_min': self.playback_rate_min,
            'slowdown_time_s': self.slowdown_time_s,
            'slowdown_integral': self.slowdown_integral,
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
