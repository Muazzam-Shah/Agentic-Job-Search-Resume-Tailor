"""
Cover Letter PDF Generator using fpdf2

Generates professional cover letters in PDF format with formatting
that matches the resume style for consistency.
"""

import os
import logging
from typing import Optional
from datetime import datetime
from fpdf import FPDF

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.cover_letter_generator import GeneratedCoverLetter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PDF Cover Letter Class
# ============================================================================

class PDFCoverLetter(FPDF):
    """Custom PDF class for cover letters."""
    
    def __init__(self):
        super().__init__(format='Letter')  # 8.5 x 11 inches
        self.set_margins(left=28.35, top=28.35, right=28.35)  # 1 inch margins = 28.35 points
        self.set_auto_page_break(auto=True, margin=28.35)
    
    def add_header_section(self, header_data):
        """Add cover letter header."""
        # Candidate contact info
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 6, header_data.candidate_name, ln=True)
        
        self.set_font('Helvetica', '', 10)
        if header_data.candidate_address:
            self.cell(0, 5, header_data.candidate_address, ln=True)
        self.cell(0, 5, f"{header_data.candidate_phone} | {header_data.candidate_email}", ln=True)
        
        # Space
        self.ln(10)
        
        # Date
        self.cell(0, 5, header_data.date, ln=True)
        
        # Space
        self.ln(10)
        
        # Recipient
        self.cell(0, 5, header_data.hiring_manager, ln=True)
        self.cell(0, 5, header_data.company_name, ln=True)
        if header_data.company_address:
            self.cell(0, 5, header_data.company_address, ln=True)
        
        # Space before salutation
        self.ln(10)
        
        # Salutation
        self.cell(0, 5, f"Dear {header_data.hiring_manager},", ln=True)
        
        # Space before body
        self.ln(8)
    
    def add_paragraph(self, text: str, spacing: float = 6):
        """Add a paragraph with proper formatting."""
        self.set_font('Helvetica', '', 11)
        # Replace Unicode characters that aren't Latin-1 compatible
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u2018', "'")  # Left single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '-')  # Em dash
        text = text.replace('\u2026', '...')  # Ellipsis
        self.multi_cell(0, spacing, text, align='L')
        self.ln(spacing)
    
    def add_closing(self, salutation: str, name: str):
        """Add closing salutation and signature."""
        self.ln(4)
        self.set_font('Helvetica', '', 11)
        self.cell(0, 6, salutation + ",", ln=True)
        self.ln(10)
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 6, name, ln=True)


# ============================================================================
# Cover Letter PDF Generator
# ============================================================================

def generate_cover_letter_pdf(
    cover_letter: GeneratedCoverLetter,
    output_dir: str = "output/cover_letters",
    filename: Optional[str] = None
) -> str:
    """
    Generate a PDF cover letter.
    
    Args:
        cover_letter: Generated cover letter data
        output_dir: Output directory for PDF
        filename: Custom filename (auto-generated if None)
    
    Returns:
        Path to generated PDF file
    """
    logger.info("Generating cover letter PDF")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        company = cover_letter.header.company_name.replace(" ", "_")
        job_title = cover_letter.metadata.get("job_title", "Position").replace(" ", "_")
        filename = f"CoverLetter_{company}_{job_title}.pdf"
    
    output_path = os.path.join(output_dir, filename)
    
    # Create PDF
    pdf = PDFCoverLetter()
    pdf.add_page()
    
    # Add header
    pdf.add_header_section(cover_letter.header)
    
    # Add opening paragraph
    pdf.add_paragraph(cover_letter.content.opening_paragraph)
    
    # Add body paragraphs
    pdf.add_paragraph(cover_letter.content.body_paragraph_1)
    pdf.add_paragraph(cover_letter.content.body_paragraph_2)
    
    if cover_letter.content.body_paragraph_3:
        pdf.add_paragraph(cover_letter.content.body_paragraph_3)
    
    # Add closing paragraph
    pdf.add_paragraph(cover_letter.content.closing_paragraph)
    
    # Add closing salutation
    pdf.add_closing(
        cover_letter.content.salutation,
        cover_letter.header.candidate_name
    )
    
    # Save PDF
    try:
        pdf.output(output_path)
        logger.info(f"Cover letter PDF saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving PDF: {e}")
        # Try with a simpler filename
        import re
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_output_path = os.path.join(output_dir, safe_filename)
        pdf.output(safe_output_path)
        logger.info(f"Cover letter PDF saved: {safe_output_path}")
        return safe_output_path


# ============================================================================
# Batch Generation
# ============================================================================

def generate_cover_letters_batch(
    cover_letters: list,
    output_dir: str = "output/cover_letters"
) -> list:
    """
    Generate multiple cover letter PDFs.
    
    Args:
        cover_letters: List of GeneratedCoverLetter objects
        output_dir: Output directory
    
    Returns:
        List of generated file paths
    """
    paths = []
    for cl in cover_letters:
        path = generate_cover_letter_pdf(cl, output_dir)
        paths.append(path)
    
    logger.info(f"Generated {len(paths)} cover letter PDFs")
    return paths
