"""
Research mode: read web pages, keep notes, and write a formatted research paper in
OpenOffice Writer.

    extract.capture_page() / parse_capture()   page -> notes.Source (UI Automation, no DOM access)
    notes.ResearchNotes                       the sources read so far (JSON in the run folder)
    compose.compose_paper()                   notes -> ir.Document (extractive, no language model)
    writer_ops.render()                       ir.Document -> editing operations
    writer_exec.WriterExecutor                operations -> keystrokes / toolbar and menu clicks by the person
    project.ResearchProject                   ties it together for main.py and the Session
"""
