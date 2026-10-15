import os

from marker.schema.document import Document

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disables a tokenizers warning

from collections import defaultdict
from typing import Annotated, Any, Dict, List, Optional, Type, Tuple, Union
import io
from contextlib import contextmanager
import tempfile

from marker.processors import BaseProcessor
from marker.services import BaseService
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.processors.llm import BaseLLMProcessor
from marker.processors.llm.llm_sectionheader import LLMSectionHeaderProcessor
from marker.providers.registry import provider_from_filepath
from marker.builders.document import DocumentBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters import BaseConverter
from marker.processors.blockquote import BlockquoteProcessor
from marker.processors.code import CodeProcessor
from marker.processors.debug import DebugProcessor
from marker.processors.document_toc import DocumentTOCProcessor
from marker.processors.equation import EquationProcessor
from marker.processors.footnote import FootnoteProcessor
from marker.processors.ignoretext import IgnoreTextProcessor
from marker.processors.line_numbers import LineNumbersProcessor
from marker.processors.list import ListProcessor
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.page_header import PageHeaderProcessor
from marker.processors.reference import ReferenceProcessor
from marker.processors.sectionheader import SectionHeaderProcessor
from marker.processors.table import TableProcessor
from marker.processors.text import TextProcessor
from marker.processors.block_relabel import BlockRelabelProcessor
from marker.processors.marginalia import MarginaliaProcessor
from marker.processors.blank_page import BlankPageProcessor
from marker.processors.llm.llm_equation import LLMEquationProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.settings import settings
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.registry import register_block_class
from marker.util import strings_to_classes
from marker.processors.llm.llm_handwriting import LLMHandwritingProcessor
from marker.services.ollama import OllamaService
from marker.processors.line_merge import LineMergeProcessor
from marker.processors.llm.llm_mathblock import LLMMathBlockProcessor
from marker.processors.llm.llm_page_correction import LLMPageCorrectionProcessor


class PdfConverter(BaseConverter):
    """
    A converter for processing and rendering PDF files into Markdown, JSON, HTML and other formats.
    """

    override_map: Annotated[
        Dict[BlockTypes, Type[Block]],
        "A mapping to override the default block classes for specific block types.",
        "The keys are `BlockTypes` enum values, representing the types of blocks,",
        "and the values are corresponding `Block` class implementations to use",
        "instead of the defaults.",
    ] = defaultdict()
    use_llm: Annotated[
        bool,
        "Enable higher quality processing with LLMs.",
    ] = False
    llm_native_pipeline: Annotated[
        bool,
        "Use native Marker geometry followed by whole-table/whole-expression LLM recognition.",
    ] = True
    page_batch_size: Annotated[
        int,
        "Number of PDF pages to recognize in each batch. Tables are joined across batch boundaries.",
    ] = 30
    mode: Annotated[
        str,
        "Conversion mode. 'balanced' uses the VLM layout model and full-page OCR;",
        "'fast' uses lightweight rf-detr/onnx detectors for layout + tables and only",
        "block-OCRs garbled/empty content. Defaults by device: 'balanced' on GPU,",
        "'fast' on CPU/MPS.",
    ] = None
    default_processors: Tuple[BaseProcessor, ...] = (
        BlockRelabelProcessor,
        LineMergeProcessor,
        BlockquoteProcessor,
        CodeProcessor,
        DocumentTOCProcessor,
        EquationProcessor,
        FootnoteProcessor,
        IgnoreTextProcessor,
        LineNumbersProcessor,
        ListProcessor,
        PageHeaderProcessor,
        MarginaliaProcessor,
        SectionHeaderProcessor,
        TableProcessor,
        LLMTableProcessor,
        LLMTableMergeProcessor,
        LLMFormProcessor,
        TextProcessor,
        LLMComplexRegionProcessor,
        LLMImageDescriptionProcessor,
        LLMEquationProcessor,
        LLMHandwritingProcessor,
        LLMMathBlockProcessor,
        LLMSectionHeaderProcessor,
        LLMPageCorrectionProcessor,
        ReferenceProcessor,
        BlankPageProcessor,
        DebugProcessor,
    )
    default_llm_service: BaseService = OllamaService

    @classmethod
    def uses_native_llm(cls, config, processor_list=None):
        """Resolve the same policy before construction (including batch setup)."""
        def setting(name):
            return config.get(f"{cls.__name__}_{name}", config.get(name, getattr(cls, name)))
        return bool(setting("use_llm") and setting("llm_native_pipeline")
                    and processor_list is None)

    def __init__(
        self,
        artifact_dict: Dict[str, Any],
        processor_list: Optional[List[str]] = None,
        renderer: str | None = None,
        llm_service: str | None = None,
        config=None,
    ):
        super().__init__(config)

        if config is None:
            config = {}

        # Resolve a device-appropriate default mode when the caller didn't choose
        # one. 'balanced' leans on the VLM for layout + full-page OCR and wants a
        # GPU; cpu/mps default to 'fast' (rf-detr/onnx layout, minimal VLM use).
        if not config.get("mode"):
            config["mode"] = (
                "balanced" if settings.TORCH_DEVICE_MODEL == "cuda" else "fast"
            )
        self.config = config
        self.mode = config["mode"]
        self.native_llm = self.uses_native_llm(config, processor_list)
        if self.page_batch_size < 1:
            raise ValueError("page_batch_size must be at least 1")

        for block_type, override_block_type in self.override_map.items():
            register_block_class(block_type, override_block_type)

        if processor_list is not None:
            processor_list = strings_to_classes(processor_list)
        else:
            processor_list = self.default_processors
        if self.native_llm:
            # Native extraction produces source structure only. One later
            # processor owns each complete logical table and math/text block;
            # old independent table/math rewrites must not run as well.
            processor_list = tuple(
                processor for processor in processor_list
                if not issubclass(processor, BaseLLMProcessor)
                and processor is not EquationProcessor
            )

        if renderer:
            renderer = strings_to_classes([renderer])[0]
        else:
            renderer = MarkdownRenderer

        # Put here so that resolve_dependencies can access it
        self.artifact_dict = artifact_dict

        if llm_service:
            llm_service_cls = strings_to_classes([llm_service])[0]
            llm_service = self.resolve_dependencies(llm_service_cls)
        elif self.use_llm:
            llm_service = self.resolve_dependencies(self.default_llm_service)

        # Inject llm service into artifact_dict so it can be picked up by processors, etc.
        self.artifact_dict["llm_service"] = llm_service
        self.llm_service = llm_service

        self.renderer = renderer

        if self.native_llm:
            original_config = self.config
            self.config = self.source_config()
            try:
                processor_list = self.initialize_processors(processor_list)
            finally:
                self.config = original_config
        else:
            processor_list = self.initialize_processors(processor_list)
        self.processor_list = processor_list

        self.native_llm_processor = None
        if self.native_llm:
            from marker.processors.llm.llm_native import NativeLLMProcessor
            original_config = self.config
            self.config = {**self.config, "use_llm": self.use_llm}
            try:
                self.native_llm_processor = self.resolve_dependencies(NativeLLMProcessor)
            finally:
                self.config = original_config

        self.layout_builder_class = LayoutBuilder
        self.page_count = None  # Track how many pages were converted

    def source_config(self):
        """Keep local layout/text extraction free of a local VLM server.

        Public force/disable OCR settings are retained in self.config for the
        cloud content stage. The source stage always reads the original layer.
        """
        config = dict(self.config)
        config.update(disable_ocr=True, force_ocr=False, mode="fast",
                      TableProcessor_disable_ocr=True,
                      TableProcessor_table_source_coordinates=True,
                      LayoutBuilder_disable_ocr=True, LayoutBuilder_mode="fast",
                      LayoutBuilder_force_ocr=False, LineBuilder_disable_ocr=True,
                      LineBuilder_mode="fast", DocumentBuilder_disable_ocr=True,
                      PdfProvider_disable_ocr=True, PdfProvider_force_ocr=False)
        return config

    @contextmanager
    def filepath_to_str(self, file_input: Union[str, io.BytesIO]):
        temp_file = None
        try:
            if isinstance(file_input, str):
                yield file_input
            else:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    if isinstance(file_input, io.BytesIO):
                        file_input.seek(0)
                        temp_file.write(file_input.getvalue())
                    else:
                        raise TypeError(
                            f"Expected str or BytesIO, got {type(file_input)}"
                        )

                yield temp_file.name
        finally:
            if temp_file is not None and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def build_document(self, filepath: str) -> Document:
        provider_cls = provider_from_filepath(filepath)
        original_config = self.config
        if self.native_llm:
            self.config = self.source_config()
        try:
            layout_builder = self.resolve_dependencies(self.layout_builder_class)
            line_builder = self.resolve_dependencies(LineBuilder)
            ocr_builder = self.resolve_dependencies(OcrBuilder)
            provider = provider_cls(filepath, self.config)
            document = DocumentBuilder(self.config)(
                provider, layout_builder, line_builder, ocr_builder
            )
        finally:
            self.config = original_config
        structure_builder_cls = self.resolve_dependencies(StructureBuilder)
        structure_builder_cls(document)

        self.run_processors(document)
        if self.native_llm_processor is not None:
            self.native_llm_processor(document)

        return document

    def run_processors(self, document: Document):
        for processor in self.processor_list:
            # LLM processors work on individual blocks/pages. Restrict each
            # invocation to a bounded set of pages, while the table merge
            # processor sees the complete document across those boundaries.
            if (
                isinstance(processor, (BaseLLMProcessor, TableProcessor))
                and not isinstance(
                    processor, (LLMTableMergeProcessor, LLMSectionHeaderProcessor)
                )
            ):
                for start in range(0, len(document.pages), self.page_batch_size):
                    batch = document.model_copy(
                        update={"pages": document.pages[start:start + self.page_batch_size]}
                    )
                    batch._page_index = None
                    processor(batch)
            else:
                processor(document)

    def __call__(self, filepath: str | io.BytesIO):
        with self.filepath_to_str(filepath) as temp_path:
            document = self.build_document(temp_path)
            self.page_count = len(document.pages)
            renderer = self.resolve_dependencies(self.renderer)
            rendered = renderer(document)
        return rendered
