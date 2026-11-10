"""Diagnostic utility to scan for, identify, and verify Thalmic Labs Myo armband BLE connectivity.

Usage:
    python scripts/scan_myo.py [--timeout 15] [--connect]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import bleak

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stream.ble_myo import (  # noqa: E402
    UUID_CONTROL_SVC,
    UUID_EMG_SVC,
)

TARGET_CONTROL_NORM = UUID_CONTROL_SVC.lower().replace("-", "")


def is_myo_device(name: str | None, local_name: str | None, service_uuids: list[str]) -> bool:
    """Check if device advertisements match Thalmic Labs Myo characteristics."""
    combined_name = f"{name or ''} {local_name or ''}".lower()
    if "myo" in combined_name:
        return True

    for s in service_uuids:
        s_norm = str(s).lower().replace("-", "")
        if s_norm == TARGET_CONTROL_NORM:
            return True

    return False


async def test_connection(address: str) -> bool:
    """Test connecting to the Myo armband and verifying GATT services."""
    print(f"\n[+] Attempting test connection to {address}...")
    t0 = time.perf_counter()
    try:
        async with bleak.BleakClient(address, timeout=10.0) as client:
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[OK] Connected in {dt:.0f} ms! State: {client.is_connected}")

            services = client.services
            print("\n[+] Discovered GATT Services:")
            has_emg = False
            has_ctrl = False
            for svc in services:
                norm_svc = str(svc.uuid).lower()
                tag = ""
                if UUID_CONTROL_SVC in norm_svc:
                    tag = " <-- [Myo Control Service]"
                    has_ctrl = True
                elif UUID_EMG_SVC in norm_svc:
                    tag = " <-- [Myo EMG Data Service]"
                    has_emg = True
                print(f"    - UUID: {svc.uuid} ({svc.description}){tag}")

            if has_ctrl and has_emg:
                print("\n[SUCCESS] Authentic Thalmic Labs Myo armband with valid EMG services!")
                print("[INFO] You can connect directly using:")
                print(f"       ./view_emg.sh --mac {address}")
                print(f"       ./run.sh --mac {address}\n")
                return True
            else:
                print("\n[WARNING] Connected, but standard Myo EMG services were not detected.")
                return False

    except Exception as exc:
        print(f"[ERROR] Failed to connect: {exc}")
        return False


async def run_scan(timeout: float, try_connect: bool) -> None:
    print("=" * 70)
    print("   THALMIC LABS MYO ARMBAND BLE SCANNER & HARDWARE DIAGNOSTIC")
    print("=" * 70)
    print(f"[*] Scanning for BLE devices (timeout: {timeout:.0f}s)...")
    print("[*] TIP: Pick up the Myo armband and shake it vigorously to wake it up.")
    print("[*]      The Thalmic logo LED on Pod 4 should pulse blue.\n")

    found_myos: list[tuple[str, str, int]] = []
    seen_addresses = set()

    def detection_callback(device, adv_data):
        if device.address in seen_addresses:
            return
        seen_addresses.add(device.address)

        name = device.name or adv_data.local_name
        svcs = adv_data.service_uuids or []

        if is_myo_device(device.name, adv_data.local_name, svcs):
            print(f" >>> [MYO FOUND] {device.address} | Name: '{name}' | RSSI: {adv_data.rssi} dBm")
            found_myos.append((device.address, name or "Myo", adv_data.rssi))

    scanner = bleak.BleakScanner(detection_callback=detection_callback)
    await scanner.start()

    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        remaining = timeout - (time.perf_counter() - start)
        msg = f"\r[*] Listening... {remaining:4.1f}s left [{len(seen_addresses)} devices heard]"
        sys.stdout.write(msg)
        sys.stdout.flush()
        await asyncio.sleep(0.5)

    await scanner.stop()
    print("\n" + "=" * 70)

    if not found_myos:
        print("\n[!] NO MYO ARMBAND DETECTED DURING SCAN.")
        print("\nHardware Troubleshooting Checklist:")
        print("  1. Armband Sleep Mode: The Myo has NO power switch. It sleeps when stationary.")
        print("     -> Pick up the armband and SHAKE IT firmly until the LED pulses blue.")
        print("  2. Battery Level:")
        print("     -> If the LED does NOT light up when shaken, the battery is depleted.")
        print("     -> Connect it to a micro-USB charger (LED should illuminate).")
        print("  3. Windows Bluetooth State:")
        print("     -> Ensure Bluetooth is ON in Windows Settings.")
        print("     -> If the Myo was previously 'Paired' in Windows Settings,")
        print("        click 'Remove Device' (the client connects directly without pairing).")
        print("  4. Exclusive Connection:")
        print("     -> Ensure official MyoConnect software or other BLE apps are closed.")
        print("=" * 70)
        return

    print(f"\n[+] Scan finished. Found {len(found_myos)} Myo device(s):")
    for addr, name, rssi in found_myos:
        print(f"    - {name} at {addr} (Signal: {rssi} dBm)")

    if try_connect and found_myos:
        best_addr = found_myos[0][0]
        await test_connection(best_addr)


def main():
    parser = argparse.ArgumentParser(description="Scan and verify Thalmic Labs Myo armband BLE.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Scan timeout in seconds (default: 15.0)")
    parser.add_argument("--connect", action="store_true", default=True, help="Attempt test connection if found")
    args = parser.parse_args()

    asyncio.run(run_scan(timeout=args.timeout, try_connect=args.connect))


if __name__ == "__main__":
    main()
