"""
Convert the CINN volatility paper from LaTeX source to PDF using reportlab.
Parses the .tex file and renders a readable, well-formatted PDF.
"""

import re
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, red, blue
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable


# ── Helpers ──────────────────────────────────────────────────────────────────

def tex_to_text(s: str) -> str:
    """Convert LaTeX markup to reportlab XML markup."""
    if not s:
        return ""

    # Remove comments
    s = re.sub(r'(?<!\\)%.*$', '', s, flags=re.MULTILINE)

    # Handle \placeholder{X} → red [X]
    s = re.sub(r'\\placeholder\{([^}]*)\}', r'<font color="red"><b>[\1]</b></font>', s)

    # Bold/italic
    s = re.sub(r'\\textbf\{([^}]*)\}', r'<b>\1</b>', s)
    s = re.sub(r'\\textit\{([^}]*)\}', r'<i>\1</i>', s)
    s = re.sub(r'\\texttt\{([^}]*)\}', r'<font face="Courier">\1</font>', s)
    s = re.sub(r'\\emph\{([^}]*)\}', r'<i>\1</i>', s)

    # Citations
    s = re.sub(r'\\citep?\{([^}]*)\}', r'[\1]', s)
    s = re.sub(r'\\citet?\{([^}]*)\}', r'[\1]', s)

    # Refs
    s = re.sub(r'\\[Cc]ref\{([^}]*)\}', r'<i>\1</i>', s)
    s = re.sub(r'\\ref\{([^}]*)\}', r'<i>\1</i>', s)
    s = re.sub(r'~', ' ', s)

    # Math mode (inline) - simplify
    def math_replace(m):
        expr = m.group(1)
        expr = expr.replace('\\sigma', 'sigma')
        expr = expr.replace('\\lambda', 'lambda')
        expr = expr.replace('\\gamma', 'gamma')
        expr = expr.replace('\\kappa', 'kappa')
        expr = expr.replace('\\alpha', 'alpha')
        expr = expr.replace('\\epsilon', 'eps')
        expr = expr.replace('\\delta', 'delta')
        expr = expr.replace('\\Delta', 'Delta')
        expr = expr.replace('\\Gamma', 'Gamma')
        expr = expr.replace('\\mathcal{A}', 'A')
        expr = expr.replace('\\mathcal{L}', 'L')
        expr = expr.replace('\\mathbb{E}', 'E')
        expr = expr.replace('\\mathbb{Q}', 'Q')
        expr = expr.replace('\\mathbb{R}', 'R')
        expr = expr.replace('\\partial', 'd')
        expr = expr.replace('\\leq', '&lt;=')
        expr = expr.replace('\\geq', '>=')
        expr = expr.replace('\\times', 'x')
        expr = expr.replace('\\cdot', '*')
        expr = expr.replace('\\to', '->')
        expr = expr.replace('\\infty', 'inf')
        expr = expr.replace('\\approx', '~')
        expr = expr.replace('\\sim', '~')
        expr = expr.replace('\\text{', '')
        expr = expr.replace('\\hat{', '')
        expr = expr.replace('\\bar{', '')
        expr = expr.replace('\\sqrt{', 'sqrt(')
        expr = expr.replace('\\ln', 'ln')
        expr = expr.replace('\\log', 'log')
        expr = expr.replace('\\max', 'max')
        expr = expr.replace('\\sup', 'sup')
        expr = expr.replace('\\limsup', 'limsup')
        expr = expr.replace('\\left', '')
        expr = expr.replace('\\right', '')
        expr = expr.replace('\\bigl', '')
        expr = expr.replace('\\bigr', '')
        expr = expr.replace('\\,', ' ')
        expr = expr.replace('\\;', ' ')
        expr = expr.replace('\\quad', '  ')
        expr = expr.replace('\\qquad', '    ')
        expr = re.sub(r'\\[a-zA-Z]+', '', expr)
        expr = expr.replace('{', '').replace('}', '')
        # Don't try to create sub/super XML tags - they break with nested font tags
        # Just use underscore notation for readability
        return f'<font face="Courier" size="9">{expr}</font>'

    s = re.sub(r'\$([^$]+)\$', math_replace, s)

    # Special chars
    s = s.replace('\\&', '&amp;')
    s = s.replace('---', '---')
    s = s.replace('--', '-')
    s = s.replace("``", '"')
    s = s.replace("''", '"')
    s = s.replace('\\%', '%')
    s = s.replace('\\$', '$')
    s = s.replace('\\_', '_')
    s = s.replace('\\#', '#')
    s = s.replace('\\textbackslash', '\\')
    s = s.replace('\\\\', '<br/>')
    s = s.replace('\\newline', '<br/>')

    # Remove remaining simple commands
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    s = re.sub(r'\\vspace\{[^}]*\}', '', s)
    s = re.sub(r'\\hspace\{[^}]*\}', '', s)
    s = re.sub(r'\\noindent\b', '', s)

    # Clean up
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def parse_table(block: str) -> list:
    """Parse a LaTeX tabular into list of lists."""
    # Find tabular content
    m = re.search(r'\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}', block, re.DOTALL)
    if not m:
        return []

    content = m.group(1)
    rows = []
    for line in content.split('\\\\'):
        line = line.strip()
        if not line or line.startswith('\\toprule') or line.startswith('\\midrule') or line.startswith('\\bottomrule'):
            continue
        line = re.sub(r'\\(toprule|midrule|bottomrule|hline|cline\{[^}]*\})', '', line)
        if not line.strip():
            continue
        cells = [tex_to_text(c.strip()) for c in line.split('&')]
        if any(c.strip() for c in cells):
            rows.append(cells)
    return rows


def parse_equation(block: str) -> str:
    """Simplify a display equation to readable text."""
    # Strip environment tags
    block = re.sub(r'\\begin\{(equation|align|align\*)\}', '', block)
    block = re.sub(r'\\end\{(equation|align|align\*)\}', '', block)
    block = re.sub(r'\\label\{[^}]*\}', '', block)

    block = block.replace('\\\\', '\n')
    lines = []
    for line in block.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Basic LaTeX math to text
        line = line.replace('\\frac{', '(').replace('}{', ')/(')
        line = re.sub(r'\\underbrace\{([^}]*)\}_\{[^}]*\}', r'\1', line)
        line = re.sub(r'\\underbrace\{', '', line)
        line = re.sub(r'\}_\{\\text\{[^}]*\}\}', '', line)
        line = line.replace('\\partial', 'd')
        line = line.replace('\\sigma', 'sigma')
        line = line.replace('\\gamma', 'gamma')
        line = line.replace('\\kappa', 'kappa')
        line = line.replace('\\alpha', 'alpha')
        line = line.replace('\\lambda', 'lambda')
        line = line.replace('\\delta', 'delta')
        line = line.replace('\\epsilon', 'eps')
        line = line.replace('\\mathcal{L}', 'L')
        line = line.replace('\\mathcal{A}', 'A')
        line = line.replace('\\mathbb{E}', 'E')
        line = line.replace('\\mathbb{Q}', 'Q')
        line = line.replace('\\hat{\\loss}', 'L_hat')
        line = line.replace('\\loss', 'L')
        line = line.replace('\\totalvar', 'w')
        line = line.replace('\\logstrike', 'k')
        line = line.replace('\\maturity', 'T')
        line = line.replace('\\density', 'g')
        line = line.replace('\\forward', 'F')
        line = line.replace('\\strike', 'K')
        line = line.replace('\\abs{', '|')
        line = line.replace('\\norm{', '||')
        line = line.replace('\\max', 'max')
        line = line.replace('\\min', 'min')
        line = line.replace('\\ln', 'ln')
        line = line.replace('\\log', 'log')
        line = line.replace('\\sqrt', 'sqrt')
        line = line.replace('\\sum', 'SUM')
        line = line.replace('\\prod', 'PROD')
        line = line.replace('\\limsup', 'limsup')
        line = line.replace('\\sup', 'sup')
        line = line.replace('\\text{', '')
        line = line.replace('\\left', '')
        line = line.replace('\\right', '')
        line = line.replace('\\bigl', '')
        line = line.replace('\\bigr', '')
        line = line.replace('\\cdot', '*')
        line = line.replace('\\times', 'x')
        line = line.replace('\\leq', '<=')
        line = line.replace('\\geq', '>=')
        line = line.replace('\\to', '->')
        line = line.replace('\\infty', 'inf')
        line = line.replace('\\quad', '    ')
        line = line.replace('\\qquad', '      ')
        line = line.replace('\\,', ' ')
        line = line.replace('\\;', ' ')
        line = re.sub(r'\\[a-zA-Z]+', '', line)
        line = line.replace('{', '').replace('}', '')
        line = line.replace('  ', ' ').strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)


# ── Main Parser ──────────────────────────────────────────────────────────────

def build_pdf(tex_path: str, pdf_path: str):
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex = f.read()

    # Extract content between \begin{document} and \end{document}
    m = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', tex, re.DOTALL)
    if not m:
        raise ValueError("No \\begin{document} found")
    body = m.group(1)

    # ── Styles ───────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'PaperTitle', parent=styles['Title'],
        fontSize=16, leading=20, spaceAfter=6, alignment=TA_CENTER,
        textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'Author', parent=styles['Normal'],
        fontSize=11, leading=14, alignment=TA_CENTER,
        spaceAfter=16, textColor=HexColor('#333333')
    ))
    styles.add(ParagraphStyle(
        'AbstractBody', parent=styles['Normal'],
        fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
        leftIndent=28, rightIndent=28, spaceAfter=4,
        textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'SectionHead', parent=styles['Heading1'],
        fontSize=14, leading=18, spaceBefore=18, spaceAfter=8,
        textColor=HexColor('#1a1a2e'), keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'SubsectionHead', parent=styles['Heading2'],
        fontSize=12, leading=15, spaceBefore=12, spaceAfter=6,
        textColor=HexColor('#2d3436'), keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'SubsubsectionHead', parent=styles['Heading3'],
        fontSize=10.5, leading=13, spaceBefore=10, spaceAfter=4,
        textColor=HexColor('#2d3436'), keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, leading=13.5, alignment=TA_JUSTIFY,
        spaceAfter=4, textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'Equation', parent=styles['Normal'],
        fontSize=9, leading=12, alignment=TA_CENTER,
        fontName='Courier', spaceAfter=8, spaceBefore=8,
        leftIndent=36, rightIndent=36,
        textColor=HexColor('#2d3436')
    ))
    styles.add(ParagraphStyle(
        'BulletItem', parent=styles['Normal'],
        fontSize=10, leading=13, alignment=TA_JUSTIFY,
        leftIndent=28, bulletIndent=14, spaceAfter=2,
        textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'EnumItem', parent=styles['Normal'],
        fontSize=10, leading=13, alignment=TA_JUSTIFY,
        leftIndent=28, bulletIndent=14, spaceAfter=2,
        textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'Caption', parent=styles['Normal'],
        fontSize=9, leading=12, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=12,
        textColor=HexColor('#555555')
    ))
    styles.add(ParagraphStyle(
        'TheoremStyle', parent=styles['Normal'],
        fontSize=10, leading=13.5, alignment=TA_JUSTIFY,
        leftIndent=14, rightIndent=14, spaceAfter=6, spaceBefore=6,
        textColor=HexColor('#1a1a2e'), backColor=HexColor('#f8f9fa'),
        borderWidth=0.5, borderColor=HexColor('#dee2e6'),
        borderPadding=6
    ))
    styles.add(ParagraphStyle(
        'KeywordsStyle', parent=styles['Normal'],
        fontSize=9, leading=12, spaceAfter=4,
        textColor=HexColor('#555555')
    ))
    styles.add(ParagraphStyle(
        'BibItem', parent=styles['Normal'],
        fontSize=8.5, leading=11, spaceAfter=3,
        leftIndent=20, firstLineIndent=-20,
        textColor=HexColor('#333333')
    ))

    story = []

    # ── Parse line by line ───────────────────────────────────────────────
    lines = body.split('\n')
    i = 0
    in_abstract = False
    in_itemize = False
    in_enumerate = False
    enum_counter = 0
    in_table = False
    table_block = ""
    in_equation = False
    eq_block = ""
    in_theorem = False
    theorem_type = ""
    theorem_block = ""
    in_biblio = False
    section_counter = 0
    subsection_counter = 0
    appendix_mode = False
    skip_until_end = None

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Skip pure comments
        if line.startswith('%'):
            continue

        # Skip \maketitle (we handle title manually)
        if line == '\\maketitle':
            # Extract title from preamble
            tm = re.search(r'\\title\{(.*?)(?:\}$)', tex, re.DOTALL)
            if tm:
                title_text = tm.group(1).replace('\\\\', '<br/>').replace('\n', ' ')
                title_text = re.sub(r'\\[a-zA-Z]+\{', '', title_text).replace('}', '')
                story.append(Paragraph(title_text, styles['PaperTitle']))

            am = re.search(r'\\author\{(.*?)\}', tex, re.DOTALL)
            if am:
                author_text = am.group(1).replace('\\\\', '<br/>').replace('\n', ' ')
                author_text = tex_to_text(author_text)
                story.append(Paragraph(author_text, styles['Author']))

            story.append(Spacer(1, 8))
            continue

        # Skip figure includes (no actual figures)
        if '\\includegraphics' in line:
            story.append(Paragraph(
                '<i>[Figure placeholder - compile with pdflatex for actual figures]</i>',
                styles['Caption']
            ))
            continue

        # Skip environments we handle specially
        if line.startswith('\\renewcommand') or line.startswith('\\renewenvironment'):
            # Skip multi-line renewenvironment
            if 'renewenvironment' in line:
                while i < len(lines) and not lines[i].strip().startswith('\\begin'):
                    i += 1
            continue

        # ── Abstract ─────────────────────────────────────────────────
        if '\\begin{abstract}' in line:
            in_abstract = True
            story.append(Paragraph('<b>Abstract</b>', styles['SectionHead']))
            continue
        if '\\end{abstract}' in line:
            in_abstract = False
            story.append(Spacer(1, 8))
            continue

        # ── Appendix ─────────────────────────────────────────────────
        if line == '\\appendix':
            appendix_mode = True
            section_counter = 0
            story.append(PageBreak())
            story.append(Paragraph('Appendices', styles['SectionHead']))
            continue

        # ── Sections ─────────────────────────────────────────────────
        sm = re.match(r'\\section\*?\{(.+?)\}', line)
        if sm:
            section_counter += 1
            subsection_counter = 0
            prefix = chr(64 + section_counter) if appendix_mode else str(section_counter)
            title = tex_to_text(sm.group(1))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f'{prefix}. {title}', styles['SectionHead']))
            continue

        ssm = re.match(r'\\subsection\*?\{(.+?)\}', line)
        if ssm:
            subsection_counter += 1
            prefix = f'{section_counter}.{subsection_counter}'
            title = tex_to_text(ssm.group(1))
            story.append(Paragraph(f'{prefix} {title}', styles['SubsectionHead']))
            continue

        sssm = re.match(r'\\subsubsection\*?\{(.+?)\}', line)
        if sssm:
            title = tex_to_text(sssm.group(1))
            story.append(Paragraph(title, styles['SubsubsectionHead']))
            continue

        # ── Theorem-like environments ────────────────────────────────
        thm_match = re.match(r'\\begin\{(definition|proposition|remark|claim|example|theorem|lemma)\}(?:\[(.+?)\])?', line)
        if thm_match:
            theorem_type = thm_match.group(1).capitalize()
            theorem_title = thm_match.group(2) or ""
            in_theorem = True
            theorem_block = ""
            continue

        if re.match(r'\\end\{(definition|proposition|remark|claim|example|theorem|lemma)\}', line):
            header = f'<b>{theorem_type}'
            if theorem_title:
                header += f' ({theorem_title})'
            header += '.</b> '
            text = tex_to_text(theorem_block)
            story.append(Paragraph(header + text, styles['TheoremStyle']))
            in_theorem = False
            theorem_block = ""
            theorem_title = ""
            continue

        if in_theorem:
            theorem_block += " " + line
            continue

        # ── Tables ───────────────────────────────────────────────────
        if '\\begin{table}' in line:
            in_table = True
            table_block = ""
            continue
        if '\\end{table}' in line:
            in_table = False
            # Parse caption
            cap_match = re.search(r'\\caption\{(.+?)\}', table_block, re.DOTALL)
            caption_text = tex_to_text(cap_match.group(1)) if cap_match else ""

            # Parse table data
            rows = parse_table(table_block)
            if rows:
                # Normalize column counts
                max_cols = max(len(r) for r in rows)
                for r in rows:
                    while len(r) < max_cols:
                        r.append("")

                # Build reportlab table
                para_rows = []
                for ri, row in enumerate(rows):
                    para_row = []
                    for cell in row:
                        style_name = 'Body'
                        cell_text = cell if cell else ""
                        try:
                            para_row.append(Paragraph(cell_text, styles[style_name]))
                        except Exception:
                            para_row.append(Paragraph(cell_text.replace('<', '&lt;').replace('>', '&gt;'), styles[style_name]))
                    para_rows.append(para_row)

                col_width = (A4[0] - 2*inch) / max_cols
                tbl = Table(para_rows, colWidths=[col_width]*max_cols)
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e8eaf6')),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('LINEBELOW', (0, 0), (-1, 0), 1, HexColor('#333333')),
                    ('LINEBELOW', (0, -1), (-1, -1), 1, HexColor('#333333')),
                ]))
                story.append(Spacer(1, 6))
                story.append(tbl)
                if caption_text:
                    story.append(Paragraph(caption_text, styles['Caption']))

            table_block = ""
            continue
        if in_table:
            table_block += line + "\n"
            continue

        # ── Equations ────────────────────────────────────────────────
        eq_match = re.match(r'\\begin\{(equation|align|align\*)\}', line)
        if eq_match:
            in_equation = True
            eq_block = ""
            continue
        if re.match(r'\\end\{(equation|align|align\*)\}', line):
            in_equation = False
            eq_text = parse_equation(eq_block)
            for eq_line in eq_text.split('\n'):
                if eq_line.strip():
                    story.append(Paragraph(eq_line.strip(), styles['Equation']))
            continue
        if in_equation:
            eq_block += line + "\n"
            continue

        # ── Lists ────────────────────────────────────────────────────
        if '\\begin{itemize}' in line:
            in_itemize = True
            continue
        if '\\end{itemize}' in line:
            in_itemize = False
            story.append(Spacer(1, 4))
            continue
        if '\\begin{enumerate}' in line:
            in_enumerate = True
            enum_counter = 0
            continue
        if '\\end{enumerate}' in line:
            in_enumerate = False
            story.append(Spacer(1, 4))
            continue

        if (in_itemize or in_enumerate) and '\\item' in line:
            item_text = re.sub(r'\\item\s*(?:\[.*?\])?\s*', '', line)
            # Collect continuation lines
            while i < len(lines):
                next_line = lines[i].strip()
                if (not next_line or next_line.startswith('\\item') or
                    next_line.startswith('\\end{') or next_line.startswith('\\begin{') or
                    next_line.startswith('\\section') or next_line.startswith('\\subsection')):
                    break
                item_text += " " + next_line
                i += 1

            item_text = tex_to_text(item_text)
            if in_enumerate:
                enum_counter += 1
                bullet = f'{enum_counter}.'
            else:
                bullet = '\u2022'

            story.append(Paragraph(
                f'{bullet}  {item_text}',
                styles['BulletItem']
            ))
            continue

        # ── Bibliography ─────────────────────────────────────────────
        if '\\begin{thebibliography}' in line:
            in_biblio = True
            story.append(Spacer(1, 8))
            story.append(Paragraph('References', styles['SectionHead']))
            continue
        if '\\end{thebibliography}' in line:
            in_biblio = False
            continue
        if in_biblio and '\\bibitem' in line:
            # Collect the full bibitem
            bib_text = re.sub(r'\\bibitem\[.*?\]\{.*?\}\s*', '', line)
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line or next_line.startswith('\\bibitem') or next_line.startswith('\\end{'):
                    break
                bib_text += " " + next_line
                i += 1
            bib_text = tex_to_text(bib_text)
            story.append(Paragraph(bib_text, styles['BibItem']))
            continue

        # ── Figures (caption only) ───────────────────────────────────
        if '\\begin{figure}' in line:
            skip_until_end = 'figure'
            continue
        if skip_until_end and f'\\end{{{skip_until_end}}}' in line:
            skip_until_end = None
            continue
        if skip_until_end:
            # Extract caption from figures
            cap_m = re.match(r'\\caption\{(.+?)\}', line)
            if cap_m:
                story.append(Paragraph(
                    '<i>[Figure] ' + tex_to_text(cap_m.group(1)) + '</i>',
                    styles['Caption']
                ))
            continue

        # ── Skip formatting commands ─────────────────────────────────
        if line.startswith('\\bibliographystyle'):
            continue
        if line.startswith('\\centering'):
            continue
        if line.startswith('\\label{'):
            continue

        # ── Keywords/Acronyms lines ──────────────────────────────────
        if line.startswith('\\textbf{Keywords') or line.startswith('\\textbf{Acronyms'):
            text = tex_to_text(line)
            story.append(Paragraph(text, styles['KeywordsStyle']))
            continue

        # ── Page breaks ──────────────────────────────────────────────
        if '\\pagebreak' in line or '\\newpage' in line or '\\clearpage' in line:
            story.append(PageBreak())
            continue

        # ── Regular paragraph text ───────────────────────────────────
        if line and not line.startswith('\\'):
            # Collect paragraph
            para = line
            while i < len(lines):
                next_line = lines[i].strip()
                if (not next_line or next_line.startswith('\\') or
                    next_line.startswith('%---')):
                    break
                para += " " + next_line
                i += 1

            text = tex_to_text(para)
            if text and len(text) > 2:
                if in_abstract:
                    story.append(Paragraph(text, styles['AbstractBody']))
                else:
                    story.append(Paragraph(text, styles['Body']))
            continue

        # Bold paragraph starters
        if line.startswith('\\textbf{'):
            para = line
            while i < len(lines):
                next_line = lines[i].strip()
                if (not next_line or next_line.startswith('\\section') or
                    next_line.startswith('\\subsection') or next_line.startswith('\\begin{') or
                    next_line.startswith('%---')):
                    break
                para += " " + next_line
                i += 1

            text = tex_to_text(para)
            if text:
                if in_abstract:
                    story.append(Paragraph(text, styles['AbstractBody']))
                else:
                    story.append(Paragraph(text, styles['Body']))
            continue

    # ── Build PDF ────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=1*inch,
        rightMargin=1*inch,
        topMargin=0.8*inch,
        bottomMargin=0.8*inch,
        title="Arbitrage-Free Volatility Surface Learning on Sparse Strike Grids",
        author="Diego Urdaneta"
    )

    # Add page numbers
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(HexColor('#888888'))
        canvas.drawCentredString(A4[0]/2, 0.5*inch, f"- {doc.page} -")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF written to: {pdf_path}")
    print(f"Pages: ~{len(story)//30 + 1} (estimated)")


if __name__ == "__main__":
    tex_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "research", "pinn_volatility_paper.tex"
    )
    pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "research", "pinn_volatility_paper.pdf"
    )
    build_pdf(tex_path, pdf_path)
