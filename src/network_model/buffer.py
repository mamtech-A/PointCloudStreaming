"""Client playback buffer with startup, rebuffer, and request-pacing clocks."""

import math


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
        
        # Fraction [0, 1) already played from the first stored frame. Keeping
        # this residual makes repeated non-frame-aligned waits and downloads
        # equivalent to one continuous playback interval.
        self._playback_frame_progress = 0.0

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
        self.total_frames_received = 0
        self.buffer_history = []  # تاریخچه سطح بافر
        self.rebuffer_count = 0  # تعداد rebuffering events

    def advance_playback(self, elapsed_time_s, now_s, event='request_pacing'):
        """Advance playback during an intentional no-request pacing wait.

        Request pacing is used only while playback has sufficient buffered
        media, so it consumes buffer without creating a stall. Updating
        ``last_update_time`` prevents the same interval from being consumed
        again when the next segment arrives.
        """
        elapsed_time_s = max(0.0, float(elapsed_time_s))
        now_s = float(now_s)
        if elapsed_time_s <= 0.0:
            return {
                'event': event, 'elapsed_time_s': 0.0,
                'buffer_level_s': self.buffer_level_s,
            }
        if not self.is_playing or self.is_rebuffering:
            raise RuntimeError(
                "request pacing requires active, non-stalled playback"
            )
        consumed = self._consume_buffer(elapsed_time_s)
        if consumed + 1e-9 < elapsed_time_s:
            raise RuntimeError(
                "request pacing exhausted the buffer; pacing threshold is invalid"
            )
        self.last_update_time = now_s
        self.buffer_history.append({
            'time_s': now_s,
            'buffer_level_s': self.buffer_level_s,
            'event': event,
            'elapsed_time_s': elapsed_time_s,
            'stall_time_s': 0.0,
        })
        return {
            'event': event,
            'elapsed_time_s': elapsed_time_s,
            'buffer_level_s': self.buffer_level_s,
        }

    def validate_atomic_segment(self, segment_frames):
        """Validate whole-segment admission for startup and rebuffer refill."""
        segment_frames = int(segment_frames)
        if segment_frames <= 0:
            raise ValueError("segment_frames must be positive")
        segment_duration_s = segment_frames / float(self.target_fps)
        if segment_duration_s > self.buffer_capacity_s + 1e-9:
            raise ValueError(
                f"segment duration {segment_duration_s:.6f}s exceeds buffer "
                f"capacity {self.buffer_capacity_s:.6f}s"
            )
        if self.min_buffer_s <= 0.0:
            return
        # Playback is paused during startup/rebuffer, so complete atomic
        # segments accumulate until the resume threshold is reached.
        segments_to_resume = int(math.ceil(
            max(0.0, self.min_buffer_s - 1e-9) / segment_duration_s
        ))
        prefill_s = segments_to_resume * segment_duration_s
        if prefill_s > self.buffer_capacity_s + 1e-9:
            raise ValueError(
                f"atomic {segment_duration_s:.6f}s segments cannot reach the "
                f"{self.min_buffer_s:.6f}s startup/rebuffer threshold within "
                f"the {self.buffer_capacity_s:.6f}s buffer; choose a smaller "
                "segment or a compatible threshold"
            )

    def _consume_buffer(self, elapsed_time_s):
        """
        مصرف بافر بر اساس زمان سپری شده (پخش فریم‌ها)
        فقط وقتی is_playing=True مصرف انجام میشه
        Returns: مقدار واقعی پخش شده
        """
        if not self.is_playing:
            return 0.0

        # Fixed-rate playback: one wall second consumes one content second.
        # The return value is the wall time actually played; callers account for
        # the unplayed remainder as rebuffering time.
        consumption_needed = elapsed_time_s

        if self.buffer_level_s + 1e-12 >= consumption_needed:
            # بافر کافی داریم - مصرف عادی
            self.buffer_level_s = max(
                0.0, self.buffer_level_s - consumption_needed
            )

            # تعداد فریم‌هایی که باید پخش بشن = زمان مصرف شده × FPS
            # ولی نباید بیشتر از فریم‌های موجود در بافر باشه
            self._playback_frame_progress += (
                consumption_needed * self.target_fps
            )
            frames_to_consume = int(self._playback_frame_progress + 1e-9)
            frames_consumed = min(frames_to_consume, len(self.frames_in_buffer))
            self.frames_played += frames_consumed
            self._playback_frame_progress = max(
                0.0, self._playback_frame_progress - frames_consumed
            )

            # حذف فریم‌ها از لیست
            for _ in range(frames_consumed):
                if self.frames_in_buffer:
                    self.frames_in_buffer.pop(0)
            if self.buffer_level_s <= 1e-12:
                # At exact exhaustion the final stored frame is complete even
                # if binary rounding left the progress infinitesimally short.
                self.frames_played += len(self.frames_in_buffer)
                self.frames_in_buffer.clear()
                self._playback_frame_progress = 0.0
            return elapsed_time_s
        else:
            # بافر کافی نیست - مصرف تا حد ممکن و سپس توقف پخش
            if self.buffer_level_s > 0:
                consumed_wall = self.buffer_level_s

                # پخش همه فریم‌های موجود در بافر
                frames_consumed = len(self.frames_in_buffer)
                self.frames_played += frames_consumed

                # خالی کردن بافر
                self.frames_in_buffer.clear()
                self.buffer_level_s = 0.0
                self._playback_frame_progress = 0.0
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
            unplayed_time = max(0.0, elapsed_time - consumed)
            
            # آیا بافر خالی شد؟
            # An empty buffer becomes a rebuffer event only after a positive
            # interval cannot be played. An immediate arrival exactly at the
            # depletion boundary preserves continuous playback.
            if self.buffer_level_s <= 1e-12 and unplayed_time > 1e-12:
                # شروع Rebuffering!
                self.is_playing = False
                self.is_rebuffering = True
                self.rebuffer_start_time = arrival_time_s - (elapsed_time - consumed) if consumed < elapsed_time else arrival_time_s
                self.rebuffer_count += 1
                # زمان stall = زمان باقیمانده که نتونستیم پخش کنیم
                stall_time = unplayed_time
                event_type = 'rebuffering_start'
                
        elif self.is_rebuffering:
            # در حال rebuffering - فقط بافر پر میشه، stall ادامه داره
            # stall time = کل زمان سپری شده در این بازه
            stall_time = elapsed_time
        
        # به‌روزرسانی زمان
        self.last_update_time = arrival_time_s
        
        # Request pacing reserves room for the complete atomic segment.
        # A complete segment must have been admitted by request pacing before
        # its atomic arrival. Overflow is therefore an invariant violation,
        # never a media-loss event or a QoE term.
        if self.buffer_level_s + self.frame_duration > self.buffer_capacity_s + 1e-9:
            raise RuntimeError(
                "segment arrival exceeds buffer capacity; request pacing was not applied"
            )
        
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
        if (not self.playback_started
                and self.buffer_level_s + 1e-9 >= self.min_buffer_s):
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
        elif (self.is_rebuffering
              and self.buffer_level_s + 1e-9 >= self.min_buffer_s):
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
                'stall_time_s': stall_time,
                'rebuffer_duration_s': rebuffer_duration,
            })
            
            return {
                'status': 'buffered',
                'event': 'playback_resumed',
                'buffer_level_s': self.buffer_level_s,
                'buffer_frames': len(self.frames_in_buffer),
                'playback_started': self.playback_started,
                'is_playing': self.is_playing,
                'frame_id': frame_id,
                # Increment accrued since the preceding buffer update. The full
                # event duration is reported separately and must not be summed
                # with prior interval deltas.
                'stall_time_s': stall_time,
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
            # Startup delay = wall-clock wait until playback FIRST started. Before
            # that moment it grows with the clock (an unstarted session is all
            # startup delay). Distinct from stalls, which only accrue after start.
            'startup_delay_s': (self.playback_start_time if self.playback_started
                                else self.last_update_time),
            'is_playing': self.is_playing,
            'is_rebuffering': self.is_rebuffering,
            'frames_played': self.frames_played,
            'frames_received': self.total_frames_received,
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
