# Checkpoint Procedure

Perform these steps when the user requests a checkpoint:

1. **Review changes since last checkpoint**
   ```bash
   git log --oneline -10
   git status
   git diff --stat
   ```

2. **Reflect on work**
   - What was accomplished?
   - What worked well? What didn't?
   - Any gotchas or lessons learnt?

3. **Update documentation**
   - Update `docs/relnotes.md` with changes made AND lessons learnt
   - Update other relevant docs if needed
   - Update CLAUDE.local.md if needed, especially for sensitive info that does not go to git (may be explicitly specified)

4. **Show changes and get approval**
   - List all files to be committed
   - Summarize the changes
   - **ASK USER FOR APPROVAL before committing**

5. **Commit all changes** (local only, no push)
   - Only proceed after user approval
   - Stage all completed work
   - Use appropriate commit message

6. **Report to user**
   - Confirm what was documented and committed
   - Summarize key lessons learnt
   - Note: Do NOT push to remote (that's for EOD)

7. **Compact Conversation**
   - Compact conversation and explcitly retain the "Report to user" part for the new context
