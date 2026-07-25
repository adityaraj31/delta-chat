# Import adapters so @register_adapter decorators fire
from src.ingest.dwg import DwgAdapter  # noqa: F401
from src.ingest.pdf_native import PdfNativeAdapter  # noqa: F401
from src.ingest.pdf_scanned import PdfScannedAdapter  # noqa: F401