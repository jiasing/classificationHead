from __future__ import annotations

from data.schema import LocalizationSample


TOY_LOCALIZATION_DATASET: list[LocalizationSample] = [
    LocalizationSample(
        sample_id="toy_001_strcpy_overflow",
        language="c",
        code=(
            "void copy_input(char *src) {\n"
            "    char buf[8];\n"
            "    strcpy(buf, src);\n"
            "    puts(buf);\n"
            "}"
        ),
        lines=[
            "void copy_input(char *src) {",
            "    char buf[8];",
            "    strcpy(buf, src);",
            "    puts(buf);",
            "}",
        ],
        line_labels=[0, 0, 1, 0, 0],
    ),
    LocalizationSample(
        sample_id="toy_002_gets_usage",
        language="c",
        code=(
            "int read_name(void) {\n"
            "    char name[16];\n"
            "    gets(name);\n"
            "    return name[0];\n"
            "}"
        ),
        lines=[
            "int read_name(void) {",
            "    char name[16];",
            "    gets(name);",
            "    return name[0];",
            "}",
        ],
        line_labels=[0, 0, 1, 0, 0],
    ),
    LocalizationSample(
        sample_id="toy_003_divide_by_length",
        language="c",
        code=(
            "int average(int sum, int count) {\n"
            "    if (sum < 0) {\n"
            "        return 0;\n"
            "    }\n"
            "    return sum / count;\n"
            "}"
        ),
        lines=[
            "int average(int sum, int count) {",
            "    if (sum < 0) {",
            "        return 0;",
            "    }",
            "    return sum / count;",
            "}",
        ],
        line_labels=[0, 0, 0, 0, 1, 0],
    ),
    LocalizationSample(
        sample_id="toy_004_null_deref",
        language="c",
        code=(
            "int read_value(struct Node *node) {\n"
            "    int value = node->value;\n"
            "    return value;\n"
            "}"
        ),
        lines=[
            "int read_value(struct Node *node) {",
            "    int value = node->value;",
            "    return value;",
            "}",
        ],
        line_labels=[0, 1, 0, 0],
    ),
    LocalizationSample(
        sample_id="toy_005_format_string",
        language="c",
        code=(
            "void log_message(char *msg) {\n"
            "    printf(msg);\n"
            "}"
        ),
        lines=[
            "void log_message(char *msg) {",
            "    printf(msg);",
            "}",
        ],
        line_labels=[0, 1, 0],
    ),
    LocalizationSample(
        sample_id="toy_006_memcpy_size",
        language="c",
        code=(
            "void copy_block(char *dst, char *src, size_t n) {\n"
            "    char local[32];\n"
            "    memcpy(local, src, n);\n"
            "    memcpy(dst, local, n);\n"
            "}"
        ),
        lines=[
            "void copy_block(char *dst, char *src, size_t n) {",
            "    char local[32];",
            "    memcpy(local, src, n);",
            "    memcpy(dst, local, n);",
            "}",
        ],
        line_labels=[0, 0, 1, 0, 0],
    ),
    LocalizationSample(
        sample_id="toy_007_off_by_one",
        language="c",
        code=(
            "void fill(int *arr, int len) {\n"
            "    for (int i = 0; i <= len; ++i) {\n"
            "        arr[i] = 0;\n"
            "    }\n"
            "}"
        ),
        lines=[
            "void fill(int *arr, int len) {",
            "    for (int i = 0; i <= len; ++i) {",
            "        arr[i] = 0;",
            "    }",
            "}",
        ],
        line_labels=[0, 1, 1, 0, 0],
    ),
    LocalizationSample(
        sample_id="toy_008_integer_overflow_alloc",
        language="c",
        code=(
            "char *make_buffer(size_t count) {\n"
            "    size_t bytes = count * 1024;\n"
            "    return malloc(bytes);\n"
            "}"
        ),
        lines=[
            "char *make_buffer(size_t count) {",
            "    size_t bytes = count * 1024;",
            "    return malloc(bytes);",
            "}",
        ],
        line_labels=[0, 1, 1, 0],
    ),
    LocalizationSample(
        sample_id="toy_009_command_injection",
        language="c",
        code=(
            "void run_command(char *user) {\n"
            "    char cmd[128];\n"
            "    sprintf(cmd, \"cat %s\", user);\n"
            "    system(cmd);\n"
            "}"
        ),
        lines=[
            "void run_command(char *user) {",
            "    char cmd[128];",
            "    sprintf(cmd, \"cat %s\", user);",
            "    system(cmd);",
            "}",
        ],
        line_labels=[0, 0, 1, 1, 0],
    ),
    LocalizationSample(
        sample_id="toy_010_safe_copy",
        language="c",
        code=(
            "void safe_copy(char *dst, const char *src, size_t size) {\n"
            "    if (size == 0) {\n"
            "        return;\n"
            "    }\n"
            "    snprintf(dst, size, \"%s\", src);\n"
            "}"
        ),
        lines=[
            "void safe_copy(char *dst, const char *src, size_t size) {",
            "    if (size == 0) {",
            "        return;",
            "    }",
            "    snprintf(dst, size, \"%s\", src);",
            "}",
        ],
        line_labels=[0, 0, 0, 0, 0, 0],
    ),
]
