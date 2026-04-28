# Sample Comparison Report: sample-comparison-report-20260428-063846

- split: `val`
- sample_count: `5`
- random_seed: `20260428`
- baseline_predictions: `artifacts\runs\baseline-20260414-144530\baseline_predictions.jsonl`
- original_system: `prompt_only_cleanup`
- qwen_system: `base_qwen_openrouter`
- sft_predictions: `artifacts\runs\eval-sft-checkpoint-20260425-021627\predictions.jsonl`
- grpo_predictions: `artifacts\runs\eval-grpo-checkpoint-20260425-024529\predictions.jsonl`

## Summary Comparison

| Metric | decompiled | original_model | qwen_via_prompt | sft_model | grpo_model |
|:---|---:|---:|---:|---:|---:|
| json_valid_rate | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| field_complete_rate | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| readability_score | 0.515 | 0.662 | 0.515 | 0.878 | 0.878 |
| naming_score | 0.200 | 0.414 | 0.000 | 0.914 | 0.914 |
| compile_success_rate | 0.000 | 0.400 | 0.000 | 0.800 | 0.800 |
| behavior_success_rate | 0.000 | 0.400 | 0.000 | 0.800 | 0.800 |

## f30d306d8716c96f

- project_id: `cfg_token_pipeline_001_v1`
- function: `count_csv_items`
- task_type: `rename`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.527 | 0.000 |
| original_model | true | false | false | 0.527 | 0.000 |
| qwen_via_prompt | false | false | false | 0.527 | 0.000 |
| sft_model | true | true | true | 0.907 | 1.000 |
| grpo_model | true | true | true | 0.907 | 1.000 |

### Original Source

```c
static int count_csv_items(const char *s) {
    int items = 0;
    int in_token = 0;
    for (; *s; s++) {
        if (*s == ',') {
            if (in_token) items++;
            in_token = 0;
        } else if (!isspace((unsigned char)*s)) {
            in_token = 1;
        }
    }
    if (in_token) items++;
    return items;
}
```

### Decompiled

```c


int __cdecl count_csv_items(char *param_1)



{

  bool bVar1;

  int iVar2;

  int local_c;

  byte *local_8;



  local_c = 0;

  bVar1 = false;

  for (local_8 = (byte *)param_1; *local_8 != 0; local_8 = local_8 + 1) {

    if (*local_8 == 0x2c) {

      if (bVar1) {

        local_c = local_c + 1;

      }

      bVar1 = false;

    }

    else {

      iVar2 = isspace((uint)*local_8);

      if (iVar2 == 0) {

        bVar1 = true;

      }

    }

  }

  if (bVar1) {

    local_c = local_c + 1;

  }

  return local_c;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for count_csv_items.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl count_csv_items(char *param_1)



{

  bool bVar1;

  int iVar2;

  int local_c;

  byte *local_8;



  local_c = 0;

  bVar1 = false;

  for (local_8 = (byte *)param_1; *local_8 != 0; local_8 = local_8 + 1) {

    if (*local_8 == 0x2c) {

      if (bVar1) {

        local_c = local_c + 1;

      }

      bVar1 = false;

    }

    else {

      iVar2 = isspace((uint)*local_8);

      if (iVar2 == 0) {

        bVar1 = true;

      }

    }

  }

  if (bVar1) {

    local_c = local_c + 1;

  }

  return local_c;

}




```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl count_csv_items(char *param_1)



{

  bool bVar1;

  int iVar2;

  int local_c;

  byte *local_8;



  local_c = 0;

  bVar1 = false;

  for (local_8 = (byte *)param_1; *local_8 != 0; local_8 = local_8 + 1) {

    if (*local_8 == 0x2c) {

      if (bVar1) {

        local_c = local_c + 1;

      }

      bVar1 = false;

    }

    else {

      iVar2 = isspace((uint)*local_8);

      if (iVar2 == 0) {

        bVar1 = true;

      }

    }

  }

  if (bVar1) {

    local_c = local_c + 1;

  }

  return local_c;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Count comma-separated nonempty items in a value string.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int count_csv_items(const char *s) {
    int count = 0;
    int in_item = 0;
    for (; *s; ++s) {
        if (*s == ',') {
            if (in_item) count++;
            in_item = 0;
        } else if (!isspace((unsigned char)*s)) {
            in_item = 1;
        }
    }
    if (in_item) count++;
    return count;
}
```

### GRPO Model

- Summary: Count comma-separated nonempty items in a value string.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int count_csv_items(const char *s) {
    int count = 0;
    int in_item = 0;
    for (; *s; ++s) {
        if (*s == ',') {
            if (in_item) count++;
            in_item = 0;
        } else if (!isspace((unsigned char)*s)) {
            in_item = 1;
        }
    }
    if (in_item) count++;
    return count;
}
```

## cac92e65b7346e34

- project_id: `cfg_token_router_001`
- function: `main`
- task_type: `full_clarify`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.515 | 1.000 |
| original_model | true | false | false | 0.516 | 1.000 |
| qwen_via_prompt | false | false | false | 0.515 | 0.000 |
| sft_model | true | false | false | 0.890 | 1.000 |
| grpo_model | true | false | false | 0.887 | 1.000 |

### Original Source

```c
int main(void) {
    char input[512];
    if (!fgets(input, sizeof input, stdin)) return 0;
    Store st = {0};
    char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    int ntok = split_tokens(input, tokens);
    if (ntok < 0) {
        puts("error=too_many_tokens");
        return 0;
    }
    if (ntok == 0) {
        puts("count=0");
        return 0;
    }
    if (!handle_command(&st, tokens, ntok)) {
        puts("error=invalid_input");
        return 0;
    }
    print_store(&st);
    return 0;
}
```

### Decompiled

```c


int __cdecl main(void)



{

  int iVar1;

  _iobuf *p_Var2;

  char *pcVar3;

  char local_858 [1032];

  Store local_450;

  char local_208 [516];

  undefined4 local_4;



  local_4 = 0;

  p_Var2 = __acrt_iob_func(0);

  pcVar3 = fgets(local_208,0x200,p_Var2);

  if (pcVar3 != (char *)0x0) {

    memset();

    iVar1 = split_tokens(local_208,(char (*) [32])local_858);

    if (iVar1 < 0) {

      puts("error=too_many_tokens");

    }

    else if (iVar1 == 0) {

      puts("count=0");

    }

    else {

      iVar1 = handle_command(&local_450,(char (*) [32])local_858,iVar1);

      if (iVar1 == 0) {

        puts("error=invalid_input");

      }

      else {

        print_store(&local_450);

      }

    }

  }

  return 0;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for main.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl main(void)



{

  int iVar1;

  _iobuf *p_Var2;

  char *pcVar3;

  char local_858 [1032];

  Store local_450;

  char local_208 [516];

  int local_4;



  local_4 = 0;

  p_Var2 = __acrt_iob_func(0);

  pcVar3 = fgets(local_208,0x200,p_Var2);

  if (pcVar3 != (char *)0x0) {

    memset();

    iVar1 = split_tokens(local_208,(char (*) [32])local_858);

    if (iVar1 < 0) {

      puts("error=too_many_tokens");

    }

    else if (iVar1 == 0) {

      puts("count=0");

    }

    else {

      iVar1 = handle_command(&local_450,(char (*) [32])local_858,iVar1);

      if (iVar1 == 0) {

        puts("error=invalid_input");

      }

      else {

        print_store(&local_450);

      }

    }

  }

  return 0;

}




```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl main(void)



{

  int iVar1;

  _iobuf *p_Var2;

  char *pcVar3;

  char local_858 [1032];

  Store local_450;

  char local_208 [516];

  undefined4 local_4;



  local_4 = 0;

  p_Var2 = __acrt_iob_func(0);

  pcVar3 = fgets(local_208,0x200,p_Var2);

  if (pcVar3 != (char *)0x0) {

    memset();

    iVar1 = split_tokens(local_208,(char (*) [32])local_858);

    if (iVar1 < 0) {

      puts("error=too_many_tokens");

    }

    else if (iVar1 == 0) {

      puts("count=0");

    }

    else {

      iVar1 = handle_command(&local_450,(char (*) [32])local_858,iVar1);

      if (iVar1 == 0) {

        puts("error=invalid_input");

      }

      else {

        print_store(&local_450);

      }

    }

  }

  return 0;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Read stdin, parse the line, execute the command, and print the final result.
- JSON valid: `True`
- Confidence: `1.000`

```c
int main(void) {
    char line[MAX_LINE];
    char *tokens[MAX_TOKENS];
    int count;
    int result;
    char *out;

    if (fgets(line, sizeof(line), stdin)) {
        memset(tokens, 0, sizeof(tokens) * MAX_TOKENS);
        count = split_tokens(line, tokens, MAX_TOKENS);
        if (count < 0) {
            puts("error=too_many_tokens");
        } else if (count == 0) {
            puts("count=0");
        } else {
            result = handle_command(tokens, count, out);
            if (result == 0) {
                puts("error=invalid_input");
            } else {
                print_store(out);
            }
        }
    }
    return 0;
}
```

### GRPO Model

- Summary: Read stdin, parse the line, execute the command, and print the final result.
- JSON valid: `True`
- Confidence: `1.000`

```c
int main(void) {
    char line[MAX_LINE];
    char *tokens[MAX_TOKENS];
    int count;
    int result;

    if (fgets(line, sizeof(line), stdin)) {
        memset(tokens, 0, sizeof(tokens) * MAX_TOKENS);
        count = split_tokens(line, tokens, MAX_TOKENS);
        if (count < 0) {
            puts("error=too_many_tokens");
        } else if (count == 0) {
            puts("count=0");
        } else {
            result = handle_command(tokens, count, line);
            if (result == 0) {
                puts("error=invalid_input");
            } else {
                print_store(&result);
            }
        }
    }
    return 0;
}
```

## 97c9bce6f13990fc

- project_id: `c11_text_flags_parser_001`
- function: `trim_inplace`
- task_type: `rename`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.516 | 0.000 |
| original_model | true | true | true | 0.884 | 0.500 |
| qwen_via_prompt | false | false | false | 0.516 | 0.000 |
| sft_model | true | true | true | 0.844 | 1.000 |
| grpo_model | true | true | true | 0.844 | 1.000 |

### Original Source

```c
static void trim_inplace(char *s) {
    size_t len = strlen(s);
    size_t start = 0;
    while (start < len && isspace((unsigned char)s[start])) start++;
    while (len > start && isspace((unsigned char)s[len - 1])) len--;
    if (start > 0) memmove(s, s + start, len - start);
    s[len - start] = '\0';
}
```

### Decompiled

```c


void __cdecl trim_inplace(char *param_1)



{

  int iVar1;

  ulonglong extraout_RAX;

  bool bVar2;

  ulonglong local_18;

  ulonglong local_10;



  strlen();

  local_18 = 0;

  while( true ) {

    bVar2 = false;

    if (local_18 < extraout_RAX) {

      iVar1 = isspace((uint)(byte)param_1[local_18]);

      bVar2 = iVar1 != 0;

    }

    local_10 = extraout_RAX;

    if (!bVar2) break;

    local_18 = local_18 + 1;

  }

  while( true ) {

    bVar2 = false;

    if (local_18 < local_10) {

      iVar1 = isspace((uint)(byte)param_1[local_10 - 1]);

      bVar2 = iVar1 != 0;

    }

    if (!bVar2) break;

    local_10 = local_10 - 1;

  }

  if (local_18 != 0) {

    memcpy();

  }

  param_1[local_10 - local_18] = '\0';

  return;

}




```

### Original Model (prompt_only_cleanup)

- Summary: This function removes leading and trailing whitespace characters from a mutable C string in place.
- JSON valid: `True`
- Confidence: `1.000`

```c
void trim_inplace(char *str) {
    size_t start = 0;
    size_t end = strlen(str);

    // Skip leading whitespace
    while (start < end && isspace((unsigned char)str[start])) {
        start++;
    }

    // Skip trailing whitespace
    while (start < end && isspace((unsigned char)str[end - 1])) {
        end--;
    }

    // Shift the trimmed string to the front if needed
    if (start > 0) {
        memcpy(str, str + start, end - start);
    }

    // Null terminate the trimmed string
    str[end - start] = '\0';
}
```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


void __cdecl trim_inplace(char *param_1)



{

  int iVar1;

  ulonglong extraout_RAX;

  bool bVar2;

  ulonglong local_18;

  ulonglong local_10;



  strlen();

  local_18 = 0;

  while( true ) {

    bVar2 = false;

    if (local_18 < extraout_RAX) {

      iVar1 = isspace((uint)(byte)param_1[local_18]);

      bVar2 = iVar1 != 0;

    }

    local_10 = extraout_RAX;

    if (!bVar2) break;

    local_18 = local_18 + 1;

  }

  while( true ) {

    bVar2 = false;

    if (local_18 < local_10) {

      iVar1 = isspace((uint)(byte)param_1[local_10 - 1]);

      bVar2 = iVar1 != 0;

    }

    if (!bVar2) break;

    local_10 = local_10 - 1;

  }

  if (local_18 != 0) {

    memcpy();

  }

  param_1[local_10 - local_18] = '\0';

  return;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Remove leading and trailing whitespace from a mutable string buffer.
- JSON valid: `True`
- Confidence: `1.000`

```c
static void trim_inplace(char *s) {
    size_t start = 0;
    size_t len = strlen(s);
    while (start < len && isspace((unsigned char)s[start])) start++;
    while (len > start && isspace((unsigned char)s[len - 1])) len--;
    if (start > 0) memmove(s, s + start, len - start);
    s[len - start] = '\0';
}
```

### GRPO Model

- Summary: Remove leading and trailing whitespace from a mutable string buffer.
- JSON valid: `True`
- Confidence: `1.000`

```c
static void trim_inplace(char *s) {
    size_t start = 0;
    size_t len = strlen(s);
    while (start < len && isspace((unsigned char)s[start])) start++;
    while (len > start && isspace((unsigned char)s[len - 1])) len--;
    if (start > 0) memmove(s, s + start, len - start);
    s[len - start] = '\0';
}
```

## 4ca167b49b3fce67

- project_id: `c11_text_config_parser_001_v2`
- function: `load_entries`
- task_type: `rename`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.496 | 0.000 |
| original_model | true | true | true | 0.863 | 0.571 |
| qwen_via_prompt | false | false | false | 0.496 | 0.000 |
| sft_model | true | true | true | 0.876 | 0.571 |
| grpo_model | true | true | true | 0.876 | 0.571 |

### Original Source

```c
static int load_entries(Entry *entries, int cap) {
    char line[MAX_LINE];
    int count = 0;
    while (fgets(line, sizeof(line), stdin)) {
        trim_inplace(line);
        if (line[0] == '\0' || line[0] == '#') continue;
        if (count >= cap) break;
        if (!parse_line(line, entries[count].key, entries[count].value)) continue;
        entries[count].flags = classify_value(entries[count].value);
        count++;
    }
    return count;
}
```

### Decompiled

```c


int __cdecl load_entries(Entry *param_1,int param_2)



{

  int iVar1;

  uint uVar2;

  _iobuf *p_Var3;

  char *pcVar4;

  int local_11c;

  char local_118 [264];

  Entry *local_10;

  int local_4;



  local_11c = 0;

  local_10 = param_1;

  local_4 = param_2;

  while( true ) {

    do {

      p_Var3 = __acrt_iob_func(0);

      pcVar4 = fgets(local_118,0x100,p_Var3);

      if (pcVar4 == (char *)0x0) {

        return local_11c;

      }

      trim_inplace(local_118);

    } while ((local_118[0] == '\0') || (local_118[0] == '#'));

    if (local_4 <= local_11c) break;

    iVar1 = parse_line(local_118,local_10[local_11c].key,local_10[local_11c].value);

    if (iVar1 != 0) {

      uVar2 = classify_value(local_10[local_11c].value);

      local_10[local_11c].flags = uVar2;

      local_11c = local_11c + 1;

    }

  }

  return local_11c;

}




```

### Original Model (prompt_only_cleanup)

- Summary: This function reads lines from standard input, skipping empty lines and lines starting with '#'. For each valid line, it parses the line into key and value fields of an Entry struct, classifies the value to set flags, and stores the Entry in an array until the maximum number of entries is reached or EOF occurs. It returns the number of successfully loaded entries.
- JSON valid: `True`
- Confidence: `0.950`

```c
int load_entries(Entry *entries, int max_entries) {
    int loaded_count = 0;
    char line_buffer[264];

    while (1) {
        FILE *stdin_file;
        char *read_result;
        int parse_result;
        unsigned int flags;

        do {
            stdin_file = __acrt_iob_func(0); // get stdin
            read_result = fgets(line_buffer, 256, stdin_file);
            if (read_result == NULL) {
                return loaded_count;
            }
            trim_inplace(line_buffer);
        } while (line_buffer[0] == '\0' || line_buffer[0] == '#');

        if (loaded_count >= max_entries) {
            break;
        }

        parse_result = parse_line(line_buffer, entries[loaded_count].key, entries[loaded_count].value);
        if (parse_result != 0) {
            flags = classify_value(entries[loaded_count].value);
            entries[loaded_count].flags = flags;
            loaded_count++;
        }
    }

    return loaded_count;
}
```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl load_entries(Entry *param_1,int param_2)



{

  int iVar1;

  uint uVar2;

  _iobuf *p_Var3;

  char *pcVar4;

  int local_11c;

  char local_118 [264];

  Entry *local_10;

  int local_4;



  local_11c = 0;

  local_10 = param_1;

  local_4 = param_2;

  while( true ) {

    do {

      p_Var3 = __acrt_iob_func(0);

      pcVar4 = fgets(local_118,0x100,p_Var3);

      if (pcVar4 == (char *)0x0) {

        return local_11c;

      }

      trim_inplace(local_118);

    } while ((local_118[0] == '\0') || (local_118[0] == '#'));

    if (local_4 <= local_11c) break;

    iVar1 = parse_line(local_118,local_10[local_11c].key,local_10[local_11c].value);

    if (iVar1 != 0) {

      uVar2 = classify_value(local_10[local_11c].value);

      local_10[local_11c].flags = uVar2;

      local_11c = local_11c + 1;

    }

  }

  return local_11c;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Read stdin, skip blank/comment lines, parse valid entries, and collect them into an array.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int load_entries(Entry *entries, int max_entries) {
    char line[MAX_LINE];
    int count = 0;
    while (fgets(line, sizeof(line), stdin)) {
        trim_inplace(line);
        if (line[0] == '\0' || line[0] == '#') continue;
        if (count >= max_entries) continue;
        Entry e;
        if (parse_line(line, e.key, e.value)) {
            e.flags = classify_value(e.value);
            entries[count++] = e;
        }
    }
    return count;
}
```

### GRPO Model

- Summary: Read stdin, skip blank/comment lines, parse valid entries, and collect them into an array.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int load_entries(Entry *entries, int max_entries) {
    char line[MAX_LINE];
    int count = 0;
    while (fgets(line, sizeof(line), stdin)) {
        trim_inplace(line);
        if (line[0] == '\0' || line[0] == '#') continue;
        if (count >= max_entries) continue;
        Entry e;
        if (parse_line(line, e.key, e.value)) {
            e.flags = classify_value(e.value);
            entries[count++] = e;
        }
    }
    return count;
}
```

## 3355a33bd0e0dbb2

- project_id: `cfg_token_router_001`
- function: `find_pair`
- task_type: `rename`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.521 | 0.000 |
| original_model | true | false | false | 0.521 | 0.000 |
| qwen_via_prompt | false | false | false | 0.521 | 0.000 |
| sft_model | true | true | true | 0.874 | 1.000 |
| grpo_model | true | true | true | 0.874 | 1.000 |

### Original Source

```c
static int find_pair(const Store *st, const char *key) {
    for (size_t i = 0; i < st->count; i++) {
        if (strcmp(st->items[i].key, key) == 0) return (int)i;
    }
    return -1;
}
```

### Decompiled

```c


int __cdecl find_pair(Store *param_1,char *param_2)



{

  int extraout_EAX;

  ulonglong local_20;



  local_20 = 0;

  while( true ) {

    if (param_1->count <= local_20) {

      return -1;

    }

    strcmp();

    if (extraout_EAX == 0) break;

    local_20 = local_20 + 1;

  }

  return (int)local_20;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for find_pair.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl find_pair(Store *param_1,char *param_2)



{

  int extraout_EAX;

  ulonglong local_20;



  local_20 = 0;

  while( true ) {

    if (param_1->count <= local_20) {

      return -1;

    }

    strcmp();

    if (extraout_EAX == 0) break;

    local_20 = local_20 + 1;

  }

  return (int)local_20;

}




```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl find_pair(Store *param_1,char *param_2)



{

  int extraout_EAX;

  ulonglong local_20;



  local_20 = 0;

  while( true ) {

    if (param_1->count <= local_20) {

      return -1;

    }

    strcmp();

    if (extraout_EAX == 0) break;

    local_20 = local_20 + 1;

  }

  return (int)local_20;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Locate a key in the in-memory key/value store.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int find_pair(const Store *st, const char *key) {
    for (size_t i = 0; i < st->count; i++) {
        if (strcmp(st->items[i].key, key) == 0) return (int)i;
    }
    return -1;
}
```

### GRPO Model

- Summary: Locate a key in the in-memory key/value store.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int find_pair(const Store *st, const char *key) {
    for (size_t i = 0; i < st->count; i++) {
        if (strcmp(st->items[i].key, key) == 0) return (int)i;
    }
    return -1;
}
```
