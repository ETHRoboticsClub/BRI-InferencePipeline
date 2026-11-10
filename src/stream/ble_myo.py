"""Bluetooth Low Energy (BLE) acquisition client for Thalmic Labs Myo armband."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
import queue
import socket
import struct
import threading
import time
from typing import Callable, Sequence

import bleak
import numpy as np

from src.stream.mock_myo import MockMyoStreamer
from src.stream.ring_buffer import EMGRingBuffer

logger = logging.getLogger(__name__)

DEFAULT_SHARED_HUB_PORT: int = 15008

UUID_CONTROL_SVC = "d5060001-a904-deb9-4748-2c7f4a124842"
UUID_COMMAND = "d5060401-a904-deb9-4748-2c7f4a124842"
UUID_EMG_SVC = "d5060005-a904-deb9-4748-2c7f4a124842"
UUID_EMG0_DATA = "d5060105-a904-deb9-4748-2c7f4a124842"
UUID_EMG1_DATA = "d5060205-a904-deb9-4748-2c7f4a124842"
UUID_EMG2_DATA = "d5060305-a904-deb9-4748-2c7f4a124842"
UUID_EMG3_DATA = "d5060405-a904-deb9-4748-2c7f4a124842"

MYO_EMG_UUIDS: tuple[str, ...] = (
    UUID_EMG0_DATA,
    UUID_EMG1_DATA,
    UUID_EMG2_DATA,
    UUID_EMG3_DATA,
)
MYO_EMG_HANDLES: tuple[int, ...] = (42, 45, 48, 51)

CMD_SET_EMG_RAW = bytearray([0x01, 0x03, 0x03, 0x00, 0x00])
CMD_SET_EMG_FILTERED = bytearray([0x01, 0x03, 0x02, 0x00, 0x00])
CMD_NEVER_SLEEP = bytearray([0x09, 0x01, 0x01])
CMD_NORMAL_SLEEP = bytearray([0x09, 0x01, 0x00])


def unpack_emg_notification(
    data: bytes | bytearray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a 16-byte Myo EMG notification into two consecutive 8-channel int8 samples."""
    raw_bytes = bytes(data)
    if len(raw_bytes) != 16:
        raise ValueError(f"Expected exactly 16 bytes for Myo EMG notification, got {len(raw_bytes)} bytes")
    s1 = np.array(struct.unpack("<8b", raw_bytes[:8]), dtype=np.float32)
    s2 = np.array(struct.unpack("<8b", raw_bytes[8:16]), dtype=np.float32)
    return s1, s2


class BleMyoClient:
    """Asynchronous Bluetooth Low Energy client for Thalmic Labs Myo armband."""

    def __init__(
        self,
        destination: (EMGRingBuffer | queue.Queue | Callable[[np.ndarray, float], None] | None) = None,
        device_address: str | None = None,
        discovery_timeout: float = 10.0,
        raw_mode: bool = True,
        mock_fallback: bool = False,
        force_mock: bool = False,
        data_path: str | pathlib.Path | None = None,
        share_connection: bool | None = None,
        share_port: int = DEFAULT_SHARED_HUB_PORT,
    ) -> None:
        self.device_address = device_address
        self.discovery_timeout = float(discovery_timeout)
        self.raw_mode = bool(raw_mode)
        self.mock_fallback = bool(mock_fallback)
        self.force_mock = bool(force_mock)
        self.data_path = data_path
        self.share_port = int(share_port)
        self.share_connection = (
            (not self.force_mock and not self.mock_fallback and self.discovery_timeout >= 1.0)
            if share_connection is None
            else bool(share_connection)
        )

        if destination is None:
            self._ring_buffer: EMGRingBuffer | None = EMGRingBuffer(capacity=2000, channels=8)
            self._destination = self._ring_buffer
        elif isinstance(destination, EMGRingBuffer):
            self._ring_buffer = destination
            self._destination = destination
        else:
            self._ring_buffer = None
            self._destination = destination

        self._is_mock: bool = self.force_mock
        self._mock_streamer: MockMyoStreamer | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: bleak.BleakClient | None = None
        self._connected: bool = False

        self._is_hub_primary: bool = False
        self._is_shared_follower: bool = False
        self._hub_server_sock: socket.socket | None = None
        self._hub_accept_thread: threading.Thread | None = None
        self._hub_clients: list[socket.socket] = []
        self._hub_lock = threading.Lock()
        self._follower_sock: socket.socket | None = None

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._is_mock

    @property
    def is_shared_follower(self) -> bool:
        return self._is_shared_follower

    @property
    def is_hub_primary(self) -> bool:
        return self._is_hub_primary

    @property
    def ring_buffer(self) -> EMGRingBuffer:
        if self._ring_buffer is not None:
            return self._ring_buffer
        if isinstance(self._destination, EMGRingBuffer):
            return self._destination
        raise AttributeError("No EMGRingBuffer attached to this BleMyoClient.")

    @property
    def buffer(self) -> EMGRingBuffer:
        return self.ring_buffer

    def _emit(self, sample: np.ndarray, timestamp: float) -> None:
        if isinstance(self._destination, EMGRingBuffer):
            self._destination.push(sample, timestamp)
        elif isinstance(self._destination, queue.Queue):
            self._destination.put((sample, timestamp))
        elif callable(self._destination):
            self._destination(sample, timestamp)

    def _broadcast_to_followers(self, frame: bytes) -> None:
        if not self._is_hub_primary:
            return
        with self._hub_lock:
            if not self._hub_clients:
                return
            dead: list[socket.socket] = []
            for cli in self._hub_clients:
                try:
                    cli.sendall(frame)
                except Exception:
                    dead.append(cli)
            for cli in dead:
                with contextlib.suppress(Exception):
                    cli.close()
                self._hub_clients.remove(cli)

    def _broadcast_status(self) -> None:
        self._broadcast_to_followers(b"S" + bytes([1 if self._connected else 0]))

    def _try_start_hub_server(self) -> bool:
        if self._is_hub_primary:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(("127.0.0.1", self.share_port))
            sock.listen(8)
            sock.settimeout(0.2)
            self._hub_server_sock = sock
            self._is_hub_primary = True
            self._is_shared_follower = False
            self._hub_accept_thread = threading.Thread(target=self._hub_accept_loop, daemon=True, name="MyoHubAccept")
            self._hub_accept_thread.start()
            return True
        except OSError:
            with contextlib.suppress(Exception):
                sock.close()
            return False

    def _hub_accept_loop(self) -> None:
        while not self._stop_event.is_set() and self._hub_server_sock is not None:
            try:
                cli, _ = self._hub_server_sock.accept()
                cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                cli.sendall(b"S" + bytes([1 if self._connected else 0]))
                with self._hub_lock:
                    self._hub_clients.append(cli)
            except socket.timeout:
                continue
            except OSError:
                break

    def _stop_hub(self) -> None:
        self._is_hub_primary = False
        self._is_shared_follower = False
        if self._follower_sock is not None:
            with contextlib.suppress(Exception):
                self._follower_sock.close()
            self._follower_sock = None
        if self._hub_server_sock is not None:
            with contextlib.suppress(Exception):
                self._hub_server_sock.close()
            self._hub_server_sock = None
        with self._hub_lock:
            for cli in self._hub_clients:
                with contextlib.suppress(Exception):
                    cli.close()
            self._hub_clients.clear()

    async def _run_shared_follower_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.5)
        try:
            await asyncio.to_thread(sock.connect, ("127.0.0.1", self.share_port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.2)
            self._follower_sock = sock
            self._is_shared_follower = True
            self._ready_event.set()

            buf = bytearray()
            while not self._stop_event.is_set():
                try:
                    chunk = await asyncio.to_thread(sock.recv, 1024)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while len(buf) >= 2:
                        mtype = buf[0:1]
                        if mtype == b"S":
                            self._connected = bool(buf[1])
                            del buf[:2]
                        elif mtype == b"E":
                            if len(buf) < 17:
                                break
                            payload = bytes(buf[1:17])
                            del buf[:17]
                            self._notification_handler(None, payload)
                        else:
                            del buf[:1]
                except socket.timeout:
                    continue
                except OSError:
                    break
        except Exception:
            pass
        finally:
            self._connected = False
            self._is_shared_follower = False
            with contextlib.suppress(Exception):
                sock.close()
            self._follower_sock = None

    def _notification_handler(self, _sender, data: bytearray | bytes) -> None:
        t_recv = time.perf_counter()
        try:
            raw_bytes = bytes(data)
            s1, s2 = unpack_emg_notification(raw_bytes)
            self._emit(s1, t_recv - 0.005)
            self._emit(s2, t_recv)
            if self._is_hub_primary:
                self._broadcast_to_followers(b"E" + raw_bytes)
        except Exception as exc:
            logger.warning("Error unpacking Myo notification: %s", exc)

    def _switch_to_mock(self) -> None:
        self._is_mock = True
        if self._mock_streamer is None:
            self._mock_streamer = MockMyoStreamer(
                destination=self._destination, sample_rate=200.0, data_source=self.data_path
            )
        self._mock_streamer.start()
        self._ready_event.set()

    async def _async_ble_run(self) -> None:
        while not self._stop_event.is_set():
            if self.share_connection and not self._is_hub_primary:
                if not self._try_start_hub_server():
                    await self._run_shared_follower_loop()
                    if self._stop_event.is_set():
                        return
                    await asyncio.sleep(0.1)
                    continue

            device = None
            try:
                if self.device_address:
                    device = await bleak.BleakScanner.find_device_by_address(
                        self.device_address, timeout=self.discovery_timeout
                    )
                else:
                    target_svc_norm = UUID_CONTROL_SVC.lower().replace("-", "")

                    def _filter(d, adv):
                        combined_name = f"{d.name or ''} {adv.local_name or ''}".lower()
                        if "myo" in combined_name or "zx05" in combined_name:
                            return True
                        if adv.service_uuids:
                            for s in adv.service_uuids:
                                if str(s).lower().replace("-", "") == target_svc_norm:
                                    return True
                        return False

                    device = await bleak.BleakScanner.find_device_by_filter(_filter, timeout=self.discovery_timeout)
            except Exception as exc:
                logger.warning("BLE scanning error: %s", exc)
                device = None

            if device is None:
                if self.mock_fallback:
                    self._switch_to_mock()
                    return
                print(
                    f"[BLE] Myo armband not found (scanned {self.discovery_timeout:.1f}s). Retrying...",
                    flush=True,
                )
                for _ in range(10):
                    if self._stop_event.is_set():
                        return
                    await asyncio.sleep(0.1)
                continue

            disconnected_event = asyncio.Event()

            def _on_disconnect(_client: bleak.BleakClient) -> None:
                self._connected = False
                self._broadcast_status()
                disconnected_event.set()
                print(f"[DISCONNECT] Myo armband disconnected ({device.address}). Reconnecting...", flush=True)

            try:
                async with bleak.BleakClient(device, disconnected_callback=_on_disconnect, timeout=10.0) as client:
                    self._client = client
                    self._connected = True
                    self._broadcast_status()
                    self._ready_event.set()

                    print(f"[CONNECTED] Myo armband ({device.address}) streaming at 200 Hz.", flush=True)

                    cmd_mode = CMD_SET_EMG_RAW if self.raw_mode else CMD_SET_EMG_FILTERED
                    await client.write_gatt_char(UUID_COMMAND, cmd_mode, response=True)
                    await client.write_gatt_char(UUID_COMMAND, CMD_NEVER_SLEEP, response=True)

                    for uuid in MYO_EMG_UUIDS:
                        await client.start_notify(uuid, self._notification_handler)

                    while not self._stop_event.is_set() and not disconnected_event.is_set():
                        await asyncio.sleep(0.05)

                    if client.is_connected:
                        for uuid in MYO_EMG_UUIDS:
                            with contextlib.suppress(Exception):
                                await client.stop_notify(uuid)
            except Exception as exc:
                self._connected = False
                self._broadcast_status()
                if not disconnected_event.is_set():
                    logger.warning("BLE connection error: %s", exc)
                if self.mock_fallback and not self._stop_event.is_set():
                    self._switch_to_mock()
                    return
            finally:
                self._connected = False
                self._broadcast_status()
                self._client = None

            if self._stop_event.is_set():
                return
            if self.mock_fallback:
                self._switch_to_mock()
                return

            for _ in range(10):
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(0.1)

    def _thread_target(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_ble_run())
        finally:
            self._stop_hub()
            self._loop.close()

    def start(self, wait_ready: bool = True, timeout: float | None = None) -> None:
        if self.is_alive():
            return
        self._stop_event.clear()
        self._ready_event.clear()

        if self.force_mock:
            self._switch_to_mock()
            return

        self._thread = threading.Thread(target=self._thread_target, daemon=True, name="BleMyoClient")
        self._thread.start()

        if wait_ready:
            max_wait = (self.discovery_timeout + 3.0) if timeout is None else float(timeout)
            self._ready_event.wait(timeout=max_wait)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._ready_event.set()
        self._stop_hub()

        if self._mock_streamer is not None:
            self._mock_streamer.stop(timeout=timeout)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        if self._is_mock and self._mock_streamer is not None:
            return self._mock_streamer.is_alive()
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> BleMyoClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
