from pydriller import Repository
import csv

# Path to the Git repository you want to analyze
repo_path = 'https://github.com/elastic/elasticsearch'

# Dictionaries to store results for each metric
file_revisions = {}  # Number of revisions per file
file_loc_changes = {}  # Lines of code added and deleted per file
performance_bugs = {}  # Performance bugs per file
non_performance_bugs = {}  # Non-performance bugs per file
file_bug_fixes = {}  # Number of revisions a file was involved in fixing bugs

# Traverse the commits in the repository
for commit in Repository(repo_path).traverse_commits():
    is_performance_bug = 'performance bug' in commit.msg.lower()
    is_bug_fix = 'fix' in commit.msg.lower() or 'bug' in commit.msg.lower()

    for modified_file in commit.modified_files:
        filename = modified_file.filename

        # 1. Number of Revisions (Commits) for Each File
        if filename not in file_revisions:
            file_revisions[filename] = 0
        file_revisions[filename] += 1

        # 2. Lines of Code Added and Deleted in File History
        if filename not in file_loc_changes:
            file_loc_changes[filename] = {'added': 0, 'deleted': 0}
        file_loc_changes[filename]['added'] += modified_file.added_lines
        file_loc_changes[filename]['deleted'] += modified_file.deleted_lines

        # 3. Counting Performance and Non-Performance Bug Fixes
        if is_performance_bug:
            if filename not in performance_bugs:
                performance_bugs[filename] = 0
            performance_bugs[filename] += 1
        elif is_bug_fix:
            if filename not in non_performance_bugs:
                non_performance_bugs[filename] = 0
            non_performance_bugs[filename] += 1

        # 4. Number of Revisions a File Was Involved in Fixing Bugs
        if is_bug_fix:
            if filename not in file_bug_fixes:
                file_bug_fixes[filename] = 0
            file_bug_fixes[filename] += 1

# Prepare data for CSV export
output_data = []
for filename in file_revisions.keys():
    output_data.append({
        'file': filename,
        'revisions': file_revisions.get(filename, 0),
        'loc_added': file_loc_changes.get(filename, {}).get('added', 0),
        'loc_deleted': file_loc_changes.get(filename, {}).get('deleted', 0),
        'performance_bug_fixes': performance_bugs.get(filename, 0),
        'non_performance_bug_fixes': non_performance_bugs.get(filename, 0),
        'bug_fix_revisions': file_bug_fixes.get(filename, 0)
    })

# Print the results
print("Results:")
for data in output_data:
    print(data)

# Export to CSV file
csv_filename = 'repository_metrics.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['file', 'revisions', 'loc_added', 'loc_deleted', 'performance_bug_fixes', 'non_performance_bug_fixes', 'bug_fix_revisions']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in output_data:
        writer.writerow(data)

print(f"\nMetrics have been saved to {csv_filename}")
