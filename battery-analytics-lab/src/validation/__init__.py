"""
Battery Analytics Lab - Validation Module
Phase 1: Data Ingestion & Standardization

This module handles validation of standardized data against quality criteria
and routes compliant/non-compliant data to appropriate directories.

Author: Battery Analytics Lab Team
Date: 2025-12-29
Version: 1.0
"""

from .data_validator import DataValidator

__all__ = ['DataValidator']