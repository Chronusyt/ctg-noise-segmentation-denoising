"""
Fetal file reader for .fetal binary format.

This module provides functionality to read .fetal files as used by the
CTG analysis system. The format supports multiple versions and multiple
fetal heart rate channels (for multiple fetuses).

[本文件为 CTG_test 自包含副本，源自 CTG/ctg_cleanCTG/ctg_io/fetal_reader.py]

File format versions:
- Version 0: Legacy format (Hong Fang Zi protocol)
- Version 1: Lian Yin protocol
- Version 2: Extended format with additional data

Each .fetal file contains:
- FHR (Fetal Heart Rate) channels (1-5 depending on number of fetuses)
- TOCO (Tocodynamometer / uterine contraction) channel
- FMP (Fetal Movement Pattern) / BFM (Binary Fetal Movement) channel
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from pathlib import Path


@dataclass
class FetalData:
    """
    Container for parsed fetal data from a .fetal file.

    Attributes:
        version: File format version (0, 1, or 2)
        fetal_num: Number of fetuses (1-5)
        fhr: Primary fetal heart rate array (unsigned 8-bit values as int)
        toco: Tocodynamometer/contraction array (unsigned 8-bit values as int)
        fmp: Fetal movement pattern array (binary 0/1 values)
        fhr2: Second fetal heart rate (for twins, None if single fetus)
        fhr3: Third fetal heart rate (None if < 3 fetuses)
        fhr4: Fourth fetal heart rate (None if < 4 fetuses)
        fhr5: Fifth fetal heart rate (None if < 5 fetuses)
        filename: Original filename
        file_path: Full path to the file
        length: Number of data points
    """
    version: int
    fetal_num: int
    fhr: np.ndarray  # int array, values 0-255
    toco: np.ndarray  # int array, values 0-255
    fmp: np.ndarray  # int array, values 0 or 1 (binary fetal movement)
    fhr2: Optional[np.ndarray] = None
    fhr3: Optional[np.ndarray] = None
    fhr4: Optional[np.ndarray] = None
    fhr5: Optional[np.ndarray] = None
    filename: str = ""
    file_path: str = ""
    length: int = 0

    def __post_init__(self):
        """Set length based on FHR array size."""
        if self.fhr is not None:
            self.length = len(self.fhr)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "fetal_num": self.fetal_num,
            "fhr": self.fhr.tolist() if self.fhr is not None else [],
            "toco": self.toco.tolist() if self.toco is not None else [],
            "fmp": self.fmp.tolist() if self.fmp is not None else [],
            "filename": self.filename,
            "file_path": self.file_path,
            "length": self.length,
        }
        if self.fhr2 is not None:
            result["fhr2"] = self.fhr2.tolist()
        if self.fhr3 is not None:
            result["fhr3"] = self.fhr3.tolist()
        if self.fhr4 is not None:
            result["fhr4"] = self.fhr4.tolist()
        if self.fhr5 is not None:
            result["fhr5"] = self.fhr5.tolist()
        return result


def _byte_to_int_hl(b: bytes) -> int:
    """
    Convert 4 bytes to integer using little-endian byte order.

    Matches Java: byteToInt_HL which reads bytes in order b[0], b[1], b[2], b[3]
    with b[0] as the least significant byte.

    Args:
        b: 4 bytes in little-endian order

    Returns:
        Integer value
    """
    return int.from_bytes(b[:4], byteorder='little', signed=False)


def read_fetal(filepath: str) -> FetalData:
    """
    Read a .fetal file and return parsed CTG data.

    This function replicates the logic of Java LmFileReader.readFetal().

    Args:
        filepath: Path to the .fetal file

    Returns:
        FetalData object containing all parsed channels

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format version is not recognized
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read entire file into bytes
    with open(filepath, 'rb') as f:
        data = f.read()

    file_len = len(data)

    # Parse header
    version = data[0] & 0xff  # Version: 0=Hong Fang Zi, 1=Lian Yin, 2=Extended
    fetal_num = data[1] & 0xff  # Number of fetuses (1-5)

    # Initialize arrays (will be set based on version)
    fhr1 = None
    fhr2 = None
    fhr3 = None
    fhr4 = None
    fhr5 = None
    toco = None
    bfm = None

    if version == 1:
        # Version 1: Lian Yin protocol
        # Each record is 8 + (fetalnum - 1) * 4 bytes
        len_every = 8 + (fetal_num - 1) * 4  # bytes per record
        num_records = (file_len - 8) // len_every

        # Initialize arrays based on fetal count
        fhr_arrays = []
        for fn in range(fetal_num):
            arr = np.zeros(num_records, dtype=np.int32)
            fhr_arrays.append(arr)
            if fn == 0:
                fhr1 = arr
            elif fn == 1:
                fhr2 = arr
            elif fn == 2:
                fhr3 = arr
            elif fn == 3:
                fhr4 = arr
            elif fn == 4:
                fhr5 = arr

        toco = np.zeros(num_records, dtype=np.int32)
        bfm = np.zeros(num_records, dtype=np.int32)

        # Parse records
        for i in range(num_records):
            base_offset = 8 + i * len_every

            # FHR1 at offset 0
            fhr1[i] = data[base_offset] & 0xff

            # BFM at offset 4 (fetal movement indicator)
            if (data[base_offset + 4] & 0xff) > 0:
                bfm[i] = 1
            else:
                bfm[i] = 0

            # TOCO at offset 5
            toco[i] = data[base_offset + 5] & 0xff

            # Additional FHR channels for multiple fetuses
            if fetal_num >= 2:
                for fn in range(1, fetal_num):
                    fhr_offset = base_offset + fn * 4 + 4
                    if fn == 1:
                        fhr2[i] = data[fhr_offset] & 0xff
                    elif fn == 2:
                        fhr3[i] = data[fhr_offset] & 0xff
                    elif fn == 3:
                        fhr4[i] = data[fhr_offset] & 0xff
                    elif fn == 4:
                        fhr5[i] = data[fhr_offset] & 0xff

    elif version == 0:
        # Version 0: Hong Fang Zi protocol (legacy)
        # File length is stored in bytes 2-5
        ifilelen = _byte_to_int_hl(data[2:6])
        doffset = fetal_num + 2  # bytes per record

        # Initialize arrays
        fhr_arrays = []
        for fn in range(fetal_num):
            arr = np.zeros(ifilelen, dtype=np.int32)
            fhr_arrays.append(arr)
            if fn == 0:
                fhr1 = arr
            elif fn == 1:
                fhr2 = arr
            elif fn == 2:
                fhr3 = arr
            elif fn == 3:
                fhr4 = arr
            elif fn == 4:
                fhr5 = arr

        toco = np.zeros(ifilelen, dtype=np.int32)
        bfm = np.zeros(ifilelen, dtype=np.int32)

        # Parse records
        for i in range(ifilelen):
            offset = 8 + i * doffset

            # Check bounds
            if offset >= file_len:
                break

            # FHR channels
            for fn in range(fetal_num):
                if fn == 0:
                    fhr1[i] = data[offset + fn] & 0xff
                elif fn == 1 and (offset + 2 + fn) < file_len:
                    fhr2[i] = data[offset + 2 + fn] & 0xff
                elif fn == 2 and (offset + 2 + fn) < file_len:
                    fhr3[i] = data[offset + 2 + fn] & 0xff
                elif fn == 3 and (offset + 2 + fn) < file_len:
                    fhr4[i] = data[offset + 2 + fn] & 0xff
                elif fn == 4 and (offset + 2 + fn) < file_len:
                    fhr5[i] = data[offset + 2 + fn] & 0xff

            # TOCO at offset + 2
            if (offset + 2) < file_len:
                toco[i] = data[offset + 2] & 0xff

            # BFM from offset + 1 (fetal movement indicator)
            if (offset + 1) < file_len:
                td1 = data[offset + 1] & 0xff
                if td1 >= 128:
                    bfm[i] = 1
                else:
                    bfm[i] = 0

    elif version == 2:
        # Version 2: Extended format
        # Similar to version 0 but with doffset = fetalnum + 3
        ifilelen = _byte_to_int_hl(data[2:6])
        doffset = fetal_num + 3  # bytes per record

        # Initialize arrays
        fhr_arrays = []
        for fn in range(fetal_num):
            arr = np.zeros(ifilelen, dtype=np.int32)
            fhr_arrays.append(arr)
            if fn == 0:
                fhr1 = arr
            elif fn == 1:
                fhr2 = arr
            elif fn == 2:
                fhr3 = arr
            elif fn == 3:
                fhr4 = arr
            elif fn == 4:
                fhr5 = arr

        toco = np.zeros(ifilelen, dtype=np.int32)
        bfm = np.zeros(ifilelen, dtype=np.int32)

        # Parse records
        for i in range(ifilelen):
            offset = 8 + i * doffset

            # Check bounds
            if offset >= file_len:
                break

            # FHR channels
            for fn in range(fetal_num):
                if fn == 0:
                    fhr1[i] = data[offset + fn] & 0xff
                elif fn == 1 and (offset + 2 + fn) < file_len:
                    fhr2[i] = data[offset + 2 + fn] & 0xff
                elif fn == 2 and (offset + 2 + fn) < file_len:
                    fhr3[i] = data[offset + 2 + fn] & 0xff
                elif fn == 3 and (offset + 2 + fn) < file_len:
                    fhr4[i] = data[offset + 2 + fn] & 0xff
                elif fn == 4 and (offset + 2 + fn) < file_len:
                    fhr5[i] = data[offset + 2 + fn] & 0xff

            # TOCO at offset + 2
            if (offset + 2) < file_len:
                toco[i] = data[offset + 2] & 0xff

            # BFM from offset + 1 (fetal movement indicator)
            if (offset + 1) < file_len:
                td1 = data[offset + 1] & 0xff
                if td1 >= 128:
                    bfm[i] = 1
                else:
                    bfm[i] = 0

    else:
        raise ValueError(f"Unknown file version: {version}. Expected 0, 1, or 2.")

    return FetalData(
        version=version,
        fetal_num=fetal_num,
        fhr=fhr1,
        toco=toco,
        fmp=bfm,
        fhr2=fhr2,
        fhr3=fhr3,
        fhr4=fhr4,
        fhr5=fhr5,
        filename=path.name,
        file_path=str(path.absolute()),
    )
