# Start of Day (SOD) Procedure

Perform the following steps to start a new development session:

1. **Re-read project context**
   - Read `CLAUDE.md` for project instructions
   - Read `docs/09-development-sop.md` for development procedures
   - Check `docs/relnotes.md` for recent changes

2. **Start dev containers**
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```

3. **Verify containers running**
   ```bash
   docker compose -f docker-compose.dev.yml ps
   ```

4. **Check git status**
   ```bash
   git status && git log --oneline -5
   ```

5. **Report to user**
   - Summarize project state
   - Note any pending work or issues
   - Confirm ready to proceed
