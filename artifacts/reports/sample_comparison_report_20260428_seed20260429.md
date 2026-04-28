# Sample Comparison Report: sample-comparison-report-20260428-063850

- split: `val`
- sample_count: `5`
- random_seed: `20260429`
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
| readability_score | 0.513 | 0.586 | 0.513 | 0.880 | 0.876 |
| naming_score | 0.200 | 0.280 | 0.000 | 1.000 | 1.000 |
| compile_success_rate | 0.000 | 0.400 | 0.000 | 0.600 | 0.800 |
| behavior_success_rate | 0.000 | 0.200 | 0.000 | 0.600 | 0.600 |

## 0d9145910347a8ac

- project_id: `cfg_token_router_001`
- function: `split_tokens`
- task_type: `rename`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.510 | 0.000 |
| original_model | true | false | false | 0.510 | 0.000 |
| qwen_via_prompt | false | false | false | 0.510 | 0.000 |
| sft_model | true | false | false | 0.910 | 1.000 |
| grpo_model | true | true | false | 0.890 | 1.000 |

### Original Source

```c
static int split_tokens(char *line, char tokens[MAX_TOKENS][MAX_TOKEN_LEN]) {
    int count = 0;
    char *p = line;
    while (*p != '\0') {
        while (*p != '\0' && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (*p == '\0') break;
        if (count >= MAX_TOKENS) return -1;
        size_t n = 0;
        while (*p != '\0' && !isspace((unsigned char)*p) && *p != ',') {
            if (n + 1 < MAX_TOKEN_LEN) tokens[count][n++] = *p;
            p++;
        }
        tokens[count][n] = '\0';
        trim_token(tokens[count]);
        count++;
    }
    return count;
}
```

### Decompiled

```c


int __cdecl split_tokens(char *param_1,char (*param_2) [32])



{

  int iVar1;

  bool bVar2;

  longlong local_30;

  byte *local_28;

  int local_1c;



  local_1c = 0;

  local_28 = (byte *)param_1;

  while( true ) {

    if (*local_28 == 0) {

      return local_1c;

    }

    while( true ) {

      bVar2 = false;

      if (*local_28 != 0) {

        iVar1 = isspace((uint)*local_28);

        bVar2 = true;

        if (iVar1 == 0) {

          bVar2 = *local_28 == 0x2c;

        }

      }

      if (!bVar2) break;

      local_28 = local_28 + 1;

    }

    if (*local_28 == 0) break;

    if (0x1f < local_1c) {

      return -1;

    }

    local_30 = 0;

    while( true ) {

      bVar2 = false;

      if (*local_28 != 0) {

        iVar1 = isspace((uint)*local_28);

        bVar2 = false;

        if (iVar1 == 0) {

          bVar2 = *local_28 != 0x2c;

        }

      }

      if (!bVar2) break;

      if (local_30 + 1U < 0x20) {

        param_2[local_1c][local_30] = *local_28;

        local_30 = local_30 + 1;

      }

      local_28 = local_28 + 1;

    }

    param_2[local_1c][local_30] = '\0';

    trim_token(param_2[local_1c]);

    local_1c = local_1c + 1;

  }

  return local_1c;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for split_tokens.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl split_tokens(char *param_1,char (*param_2) [32])



{

  int iVar1;

  bool bVar2;

  longlong local_30;

  byte *local_28;

  int local_1c;



  local_1c = 0;

  local_28 = (byte *)param_1;

  while( true ) {

    if (*local_28 == 0) {

      return local_1c;

    }

    while( true ) {

      bVar2 = false;

      if (*local_28 != 0) {

        iVar1 = isspace((uint)*local_28);

        bVar2 = true;

        if (iVar1 == 0) {

          bVar2 = *local_28 == 0x2c;

        }

      }

      if (!bVar2) break;

      local_28 = local_28 + 1;

    }

    if (*local_28 == 0) break;

    if (0x1f < local_1c) {

      return -1;

    }

    local_30 = 0;

    while( true ) {

      bVar2 = false;

      if (*local_28 != 0) {

        iVar1 = isspace((uint)*local_28);

        bVar2 = false;

        if (iVar1 == 0) {

          bVar2 = *local_28 != 0x2c;

        }

      }

      if (!bVar2) break;

      if (local_30 + 1U < 0x20) {

        param_2[local_1c][local_30] = *local_28;

        local_30 = local_30 + 1;

      }

      local_28 = local_28 + 1;

    }

    param_2[local_1c][local_30] = '\0';

    trim_token(param_2[local_1c]);

    local_1c = local_1c + 1;

  }

  return local_1c;

}




```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl split_tokens(char *param_1,char (*param_2) [32])



{

  int iVar1;

  bool bVar2;

  longlong local_30;

  byte *local_28;

  int local_1c;



  local_1c = 0;

  local_28 = (byte *)param_1;

  while( true ) {

    if (*local_28 == 0) {

      return local_1c;

    }

    while( true ) {

      bVar2 = false;

      if (*local_28 != 0) {

        iVar1 = isspace((uint)*local_28);

        bVar2 = true;

        if (iVar1 == 0) {

          bVar2 = *local_28 == 0x2c;

        }

      }

      if (!bVar2) break;

      local_28 = local_28 + 1;

    }

    if (*local_28 == 0) break;

    if (0x1f < local_1c) {

      return -1;

    }

    local_30 = 0;

    while( true ) {

      bVar2 = false;

      if (*local_28 != 0) {

        iVar1 = isspace((uint)*local_28);

        bVar2 = false;

        if (iVar1 == 0) {

          bVar2 = *local_28 != 0x2c;

        }

      }

      if (!bVar2) break;

      if (local_30 + 1U < 0x20) {

        param_2[local_1c][local_30] = *local_28;

        local_30 = local_30 + 1;

      }

      local_28 = local_28 + 1;

    }

    param_2[local_1c][local_30] = '\0';

    trim_token(param_2[local_1c]);

    local_1c = local_1c + 1;

  }

  return local_1c;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Tokenize a line using spaces and commas as separators.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int split_tokens(char *line, char **tokens) {
    size_t i = 0;
    char *p = line;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p && *p != ',') {
            tokens[i++] = *p;
        }
        p++;
    }
    tokens[i][0] = '\0';
    return i;
}
```

### GRPO Model

- Summary: Tokenize a line using spaces and commas as separators.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int split_tokens(char *line, char **tokens) {
    char *p = line;
    int count = 0;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p && *p != ',' && *p != '\n') {
            tokens[count++] = p;
            while (*p && isspace((unsigned char)*p)) p++;
        }
        if (*p) p++;
    }
    return count;
}
```

## 888b19c4b8e05c44

- project_id: `c11_text_config_parser_001_v2`
- function: `main`
- task_type: `cleanup`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.505 | 1.000 |
| original_model | true | true | false | 0.506 | 1.000 |
| qwen_via_prompt | false | false | false | 0.505 | 0.000 |
| sft_model | true | false | false | 0.840 | 1.000 |
| grpo_model | true | false | false | 0.840 | 1.000 |

### Original Source

```c
int main(void) {
    Entry entries[MAX_ENTRIES];
    int n = load_entries(entries, MAX_ENTRIES);
    int enabled = count_flag(entries, n, FLAG_ENABLED);
    int readonly = count_flag(entries, n, FLAG_READONLY);
    int numeric = count_flag(entries, n, FLAG_NUMERIC);

    printf("entries=%d\n", n);
    printf("enabled=%d\n", enabled);
    printf("readonly=%d\n", readonly);
    printf("numeric=%d\n", numeric);
    for (int i = 0; i < n; i++) {
        printf("%s:%s:%c%c%c\n",
               entries[i].key,
               entries[i].value,
               (entries[i].flags & FLAG_ENABLED) ? 'E' : '-',
               (entries[i].flags & FLAG_READONLY) ? 'R' : '-',
               (entries[i].flags & FLAG_NUMERIC) ? 'N' : '-');
    }
    return 0;
}
```

### Decompiled

```c


int __cdecl main(void)



{

  int iVar1;

  int local_65c;

  Entry local_648 [16];

  undefined4 local_4;



  local_4 = 0;

  iVar1 = load_entries(local_648,0x10);

  count_flag(local_648,iVar1,1);

  count_flag(local_648,iVar1,2);

  count_flag(local_648,iVar1,4);

  printf("entries=%d\n");

  printf("enabled=%d\n");

  printf("readonly=%d\n");

  printf("numeric=%d\n");

  for (local_65c = 0; local_65c < iVar1; local_65c = local_65c + 1) {

    printf("%s:%s:%c%c%c\n");

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

  int local_65c;

  Entry local_648 [16];

  int local_4;



  local_4 = 0;

  iVar1 = load_entries(local_648,0x10);

  count_flag(local_648,iVar1,1);

  count_flag(local_648,iVar1,2);

  count_flag(local_648,iVar1,4);

  printf("entries=%d\n");

  printf("enabled=%d\n");

  printf("readonly=%d\n");

  printf("numeric=%d\n");

  for (local_65c = 0; local_65c < iVar1; local_65c = local_65c + 1) {

    printf("%s:%s:%c%c%c\n");

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

  int local_65c;

  Entry local_648 [16];

  undefined4 local_4;



  local_4 = 0;

  iVar1 = load_entries(local_648,0x10);

  count_flag(local_648,iVar1,1);

  count_flag(local_648,iVar1,2);

  count_flag(local_648,iVar1,4);

  printf("entries=%d\n");

  printf("enabled=%d\n");

  printf("readonly=%d\n");

  printf("numeric=%d\n");

  for (local_65c = 0; local_65c < iVar1; local_65c = local_65c + 1) {

    printf("%s:%s:%c%c%c\n");

  }

  return 0;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Parse simple key=value configuration input, classify values, and print normalized summary statistics and per-entry flags.
- JSON valid: `True`
- Confidence: `1.000`

```c
int main(void) {
    Entry entries[MAX_ENTRIES];
    int count = load_entries(entries, MAX_ENTRIES);
    int enabled = count_flag(entries, count, 1);
    int readonly = count_flag(entries, count, 2);
    int numeric = count_flag(entries, count, 4);
    printf("entries=%d\n", count);
    printf("enabled=%d\n", enabled);
    printf("readonly=%d\n", readonly);
    printf("numeric=%d\n", numeric);
    for (int i = 0; i < count; i++) {
        printf("%s:%s:%c%c%c\n", entries[i].key, entries[i].value, entries[i].flag & 1u, entries[i].flag & 2u, entries[i].flag & 4u);
    }
    return 0;
}
```

### GRPO Model

- Summary: Parse simple key=value configuration input, classify values, and print normalized summary statistics and per-entry flags.
- JSON valid: `True`
- Confidence: `1.000`

```c
int main(void) {
    Entry entries[MAX_ENTRIES];
    int count = load_entries(entries, MAX_ENTRIES);
    int enabled = count_flag(entries, count, 1);
    int readonly = count_flag(entries, count, 2);
    int numeric = count_flag(entries, count, 4);
    printf("entries=%d\n", count);
    printf("enabled=%d\n", enabled);
    printf("readonly=%d\n", readonly);
    printf("numeric=%d\n", numeric);
    for (int i = 0; i < count; i++) {
        printf("%s:%s:%c%c%c\n", entries[i].key, entries[i].value, entries[i].flag & 1u, entries[i].flag & 2u, entries[i].flag & 4u);
    }
    return 0;
}
```

## d4ab1ae917416dc0

- project_id: `c_token_config_parser`
- function: `parse_hex8`
- task_type: `full_clarify`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.512 | 0.000 |
| original_model | true | true | true | 0.873 | 0.398 |
| qwen_via_prompt | false | false | false | 0.512 | 0.000 |
| sft_model | true | true | true | 0.890 | 1.000 |
| grpo_model | true | true | true | 0.890 | 1.000 |

### Original Source

```c
static int parse_hex8(const char *s, uint8_t *out) {
    unsigned int value = 0;
    if (strlen(s) != 2) return 0;
    for (int i = 0; i < 2; ++i) {
        char c = s[i];
        unsigned int digit;
        if (c >= '0' && c <= '9') digit = (unsigned int)(c - '0');
        else if (c >= 'a' && c <= 'f') digit = 10u + (unsigned int)(c - 'a');
        else if (c >= 'A' && c <= 'F') digit = 10u + (unsigned int)(c - 'A');
        else return 0;
        value = (value << 4) | digit;
    }
    *out = (uint8_t)value;
    return 1;
}
```

### Decompiled

```c


int __cdecl parse_hex8(char *param_1,uchar *param_2)



{

  char cVar1;

  longlong extraout_RAX;

  uint local_28;

  int local_20;

  uint local_1c;

  int local_4;



  local_1c = 0;

  strlen();

  if (extraout_RAX == 2) {

    for (local_20 = 0; local_20 < 2; local_20 = local_20 + 1) {

      cVar1 = param_1[local_20];

      if ((cVar1 < '0') || ('9' < cVar1)) {

        if ((cVar1 < 'a') || ('f' < cVar1)) {

          if ((cVar1 < 'A') || ('F' < cVar1)) {

            return 0;

          }

          local_28 = (int)cVar1 - 0x37;

        }

        else {

          local_28 = (int)cVar1 - 0x57;

        }

      }

      else {

        local_28 = (int)cVar1 - 0x30;

      }

      local_1c = local_1c << 4 | local_28;

    }

    *param_2 = (uchar)local_1c;

    local_4 = 1;

  }

  else {

    local_4 = 0;

  }

  return local_4;

}




```

### Original Model (prompt_only_cleanup)

- Summary: This function parses a two-character hexadecimal string from the input char pointer and converts it into an 8-bit unsigned integer. It first checks if the input string length is exactly 2. Then, for each character, it validates if it is a valid hex digit (0-9, a-f, A-F), converts it to its numeric value, and accumulates the result by shifting the previous value 4 bits left and OR-ing the new digit. If any character is invalid or the length is not 2, the function returns 0 indicating failure. On success, it stores the parsed byte in the output pointer and returns 1.
- JSON valid: `True`
- Confidence: `1.000`

```c
int parse_hex8(char *hex_str, unsigned char *out_byte) {
    unsigned int value = 0;
    if (strlen(hex_str) == 2) {
        for (int i = 0; i < 2; i++) {
            char c = hex_str[i];
            unsigned int digit;
            if (c >= '0' && c <= '9') {
                digit = c - '0';
            } else if (c >= 'a' && c <= 'f') {
                digit = c - 'a' + 10;
            } else if (c >= 'A' && c <= 'F') {
                digit = c - 'A' + 10;
            } else {
                return 0; // invalid hex digit
            }
            value = (value << 4) | digit;
        }
        *out_byte = (unsigned char)value;
        return 1; // success
    } else {
        return 0; // invalid length
    }
}
```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl parse_hex8(char *param_1,uchar *param_2)



{

  char cVar1;

  longlong extraout_RAX;

  uint local_28;

  int local_20;

  uint local_1c;

  int local_4;



  local_1c = 0;

  strlen();

  if (extraout_RAX == 2) {

    for (local_20 = 0; local_20 < 2; local_20 = local_20 + 1) {

      cVar1 = param_1[local_20];

      if ((cVar1 < '0') || ('9' < cVar1)) {

        if ((cVar1 < 'a') || ('f' < cVar1)) {

          if ((cVar1 < 'A') || ('F' < cVar1)) {

            return 0;

          }

          local_28 = (int)cVar1 - 0x37;

        }

        else {

          local_28 = (int)cVar1 - 0x57;

        }

      }

      else {

        local_28 = (int)cVar1 - 0x30;

      }

      local_1c = local_1c << 4 | local_28;

    }

    *param_2 = (uchar)local_1c;

    local_4 = 1;

  }

  else {

    local_4 = 0;

  }

  return local_4;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Convert a two-character hexadecimal string into an 8-bit value.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int parse_hex8(const char *s, uint8_t *out) {
    uint32_t value = 0;
    size_t i = 0;
    while (i < strlen(s)) {
        char c = s[i];
        if (c < '0' || c > '9') continue;
        if (c < 'a' || c > 'f') continue;
        if (c < 'A' || c > 'F') continue;
        value = (value << 4) | (uint32_t)c - '0';
        i++;
    }
    *out = (uint8_t)value;
    return 1;
}
```

### GRPO Model

- Summary: Convert a two-character hexadecimal string into an 8-bit value.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int parse_hex8(const char *s, uint8_t *out) {
    uint32_t value = 0;
    size_t i = 0;
    while (i < strlen(s)) {
        char c = s[i];
        if (c < '0' || c > '9') continue;
        if (c < 'a' || c > 'f') continue;
        if (c < 'A' || c > 'F') continue;
        value = (value << 4) | (uint32_t)c - '0';
        i++;
    }
    *out = (uint8_t)value;
    return 1;
}
```

## 833a2dc218bd6f36

- project_id: `cfg_token_pipeline_001_v1`
- function: `is_integer_token`
- task_type: `cleanup`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.520 | 0.000 |
| original_model | true | false | false | 0.520 | 0.000 |
| qwen_via_prompt | false | false | false | 0.520 | 0.000 |
| sft_model | true | true | true | 0.891 | 1.000 |
| grpo_model | true | true | true | 0.891 | 1.000 |

### Original Source

```c
static int is_integer_token(const char *s) {
    if (*s == '\0') return 0;
    if (*s == '-' || *s == '+') s++;
    if (*s == '\0') return 0;
    for (; *s; s++) {
        if (!isdigit((unsigned char)*s)) return 0;
    }
    return 1;
}
```

### Decompiled

```c


int __cdecl is_integer_token(char *param_1)



{

  int iVar1;

  byte *local_10;

  int local_4;



  if (*param_1 == '\0') {

    local_4 = 0;

  }

  else {

    if ((*param_1 == '-') || (local_10 = (byte *)param_1, *param_1 == '+')) {

      local_10 = (byte *)(param_1 + 1);

    }

    if (*local_10 == 0) {

      local_4 = 0;

    }

    else {

      for (; *local_10 != 0; local_10 = local_10 + 1) {

        iVar1 = isdigit((uint)*local_10);

        if (iVar1 == 0) {

          return 0;

        }

      }

      local_4 = 1;

    }

  }

  return local_4;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for is_integer_token.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl is_integer_token(char *param_1)



{

  int iVar1;

  byte *local_10;

  int local_4;



  if (*param_1 == '\0') {

    local_4 = 0;

  }

  else {

    if ((*param_1 == '-') || (local_10 = (byte *)param_1, *param_1 == '+')) {

      local_10 = (byte *)(param_1 + 1);

    }

    if (*local_10 == 0) {

      local_4 = 0;

    }

    else {

      for (; *local_10 != 0; local_10 = local_10 + 1) {

        iVar1 = isdigit((uint)*local_10);

        if (iVar1 == 0) {

          return 0;

        }

      }

      local_4 = 1;

    }

  }

  return local_4;

}




```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl is_integer_token(char *param_1)



{

  int iVar1;

  byte *local_10;

  int local_4;



  if (*param_1 == '\0') {

    local_4 = 0;

  }

  else {

    if ((*param_1 == '-') || (local_10 = (byte *)param_1, *param_1 == '+')) {

      local_10 = (byte *)(param_1 + 1);

    }

    if (*local_10 == 0) {

      local_4 = 0;

    }

    else {

      for (; *local_10 != 0; local_10 = local_10 + 1) {

        iVar1 = isdigit((uint)*local_10);

        if (iVar1 == 0) {

          return 0;

        }

      }

      local_4 = 1;

    }

  }

  return local_4;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Check whether a value is a signed decimal integer token.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int is_integer_token(const char *s) {
    if (s[0] == '\0') return 0;
    if (s[0] == '-' || s[0] == '+') s++;
    if (s[0] == '\0') return 0;
    for (; *s; s++) {
        if (!isdigit((unsigned char)*s)) return 0;
    }
    return 1;
}
```

### GRPO Model

- Summary: Check whether a value is a signed decimal integer token.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int is_integer_token(const char *s) {
    if (s[0] == '\0') return 0;
    if (s[0] == '-' || s[0] == '+') s++;
    if (s[0] == '\0') return 0;
    for (; *s; s++) {
        if (!isdigit((unsigned char)*s)) return 0;
    }
    return 1;
}
```

## 4ca7cb4da54c4a51

- project_id: `cfg_token_pipeline_001_v1`
- function: `is_integer_token`
- task_type: `rename`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.520 | 0.000 |
| original_model | true | false | false | 0.520 | 0.000 |
| qwen_via_prompt | false | false | false | 0.520 | 0.000 |
| sft_model | true | true | true | 0.868 | 1.000 |
| grpo_model | true | true | true | 0.868 | 1.000 |

### Original Source

```c
static int is_integer_token(const char *s) {
    if (*s == '\0') return 0;
    if (*s == '-' || *s == '+') s++;
    if (*s == '\0') return 0;
    for (; *s; s++) {
        if (!isdigit((unsigned char)*s)) return 0;
    }
    return 1;
}
```

### Decompiled

```c


int __cdecl is_integer_token(char *param_1)



{

  int iVar1;

  byte *local_10;

  int local_4;



  if (*param_1 == '\0') {

    local_4 = 0;

  }

  else {

    if ((*param_1 == '-') || (local_10 = (byte *)param_1, *param_1 == '+')) {

      local_10 = (byte *)(param_1 + 1);

    }

    if (*local_10 == 0) {

      local_4 = 0;

    }

    else {

      for (; *local_10 != 0; local_10 = local_10 + 1) {

        iVar1 = isdigit((uint)*local_10);

        if (iVar1 == 0) {

          return 0;

        }

      }

      local_4 = 1;

    }

  }

  return local_4;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for is_integer_token.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl is_integer_token(char *param_1)



{

  int iVar1;

  byte *local_10;

  int local_4;



  if (*param_1 == '\0') {

    local_4 = 0;

  }

  else {

    if ((*param_1 == '-') || (local_10 = (byte *)param_1, *param_1 == '+')) {

      local_10 = (byte *)(param_1 + 1);

    }

    if (*local_10 == 0) {

      local_4 = 0;

    }

    else {

      for (; *local_10 != 0; local_10 = local_10 + 1) {

        iVar1 = isdigit((uint)*local_10);

        if (iVar1 == 0) {

          return 0;

        }

      }

      local_4 = 1;

    }

  }

  return local_4;

}




```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl is_integer_token(char *param_1)



{

  int iVar1;

  byte *local_10;

  int local_4;



  if (*param_1 == '\0') {

    local_4 = 0;

  }

  else {

    if ((*param_1 == '-') || (local_10 = (byte *)param_1, *param_1 == '+')) {

      local_10 = (byte *)(param_1 + 1);

    }

    if (*local_10 == 0) {

      local_4 = 0;

    }

    else {

      for (; *local_10 != 0; local_10 = local_10 + 1) {

        iVar1 = isdigit((uint)*local_10);

        if (iVar1 == 0) {

          return 0;

        }

      }

      local_4 = 1;

    }

  }

  return local_4;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Check whether a value is a signed decimal integer token.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int is_integer_token(const char *s) {
    if (s[0] == '\0') return 0;
    if (s[0] == '-' || s[0] == '+') s++;
    if (s[0] == '\0') return 0;
    for (; *s; s++) if (!isdigit((unsigned char)*s)) return 0;
    return 1;
}
```

### GRPO Model

- Summary: Check whether a value is a signed decimal integer token.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int is_integer_token(const char *s) {
    if (s[0] == '\0') return 0;
    if (s[0] == '-' || s[0] == '+') s++;
    if (s[0] == '\0') return 0;
    for (; *s; s++) if (!isdigit((unsigned char)*s)) return 0;
    return 1;
}
```
