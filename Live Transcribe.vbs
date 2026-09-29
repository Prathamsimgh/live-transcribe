' Live Transcribe -- one-click launcher, no console window.
' Double-click this file, or create a desktop shortcut to it.
'
' Uses pythonw.exe so no black console flashes on screen. The venv keeps the
' dependencies out of the system Python, so this works without an install step.

Set sh = CreateObject("Wscript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = fso.BuildPath(here, ".venv\Scripts\pythonw.exe")

If Not fso.FileExists(pyw) Then
    sh.Popup "The virtual environment is missing." & vbCrLf & vbCrLf & _
             "Expected it at:" & vbCrLf & pyw & vbCrLf & vbCrLf & _
             "Create it with:" & vbCrLf & _
             "python -m venv .venv" & vbCrLf & _
             ".venv\Scripts\pip install -r requirements.txt", _
             20, "Live Transcribe", 48
    WScript.Quit 1
End If

sh.CurrentDirectory = here
sh.Run """" & pyw & """ live_transcribe.py", 0, False
