# End of Day (EOD) Procedure

Perform these steps when ending a development session:

1. **Reflect on the session**
   - What was accomplished?
   - What challenges were encountered?
   - What lessons were learnt? Any gotchas or discoveries?

2. **Review all uncommitted changes**
   ```bash
   git status
   git diff --stat
   ```

3. **Update documentation with reflections**
   - Update `docs/relnotes.md` with new features AND discoveries
   - Add lessons learnt to appropriate docs
   - Check if `docs/*.md` needs updates

4. **Update CLAUDE.md if needed**
   - Add new important concepts discovered
   - Document any new conventions or patterns
   - Add to Technical Debt section if issues need future investigation

5. **Show changes and get approval**
   - List all files to be committed
   - Summarize the changes
   - **ASK USER FOR APPROVAL before committing**

6. **Commit all changes**
   - Only proceed after user approval
   - Stage and commit all completed work with appropriate messages

7. **Push to remote** (REQUIRED for EOD)
   ```bash
   git push origin <current-branch>
   ```

8. **Stop dev containers**
   ```bash
   docker compose -f docker-compose.dev.yml down
   ```

9. **Report to user**
   - Summarize session accomplishments
   - Confirm push and container shutdown
