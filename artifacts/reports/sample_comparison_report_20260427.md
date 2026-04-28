# Sample Comparison Report: sample-comparison-report-20260427-204956

- split: `val`
- sample_count: `5`
- random_seed: `20260427`
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
| readability_score | 0.512 | 0.657 | 0.512 | 0.867 | 0.873 |
| naming_score | 0.200 | 0.300 | 0.000 | 0.867 | 0.867 |
| compile_success_rate | 0.000 | 0.200 | 0.000 | 0.600 | 0.600 |
| behavior_success_rate | 0.000 | 0.200 | 0.000 | 0.600 | 0.600 |

## 15f76cf6deb59214

- project_id: `c11_text_flags_parser_001`
- function: `trim_inplace`
- task_type: `full_clarify`
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

- Summary: This function trims leading and trailing whitespace characters from a mutable C string in place. It first calculates the string length, then advances a start index past all leading whitespace, and reduces an end index before all trailing whitespace. If any leading whitespace was removed, it shifts the remaining substring to the front of the buffer using memcpy. Finally, it null-terminates the string after the trimmed content.
- JSON valid: `True`
- Confidence: `1.000`

```c
void trim_inplace(char *str) {
    size_t len = strlen(str);
    size_t start = 0;
    size_t end = len;

    // Skip leading whitespace
    while (start < len && isspace((unsigned char)str[start])) {
        start++;
    }

    // Skip trailing whitespace
    while (start < end && isspace((unsigned char)str[end - 1])) {
        end--;
    }

    // If there was leading whitespace, shift the string left
    if (start > 0) {
        memcpy(str, str + start, end - start);
    }

    // Null-terminate the trimmed string
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

## 66690e9221bcc1e4

- project_id: `cfg_token_pipeline_001_v1`
- function: `find_pair`
- task_type: `full_clarify`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.523 | 0.000 |
| original_model | true | false | false | 0.523 | 0.000 |
| qwen_via_prompt | false | false | false | 0.523 | 0.000 |
| sft_model | true | true | true | 0.880 | 1.000 |
| grpo_model | true | true | true | 0.880 | 1.000 |

### Original Source

```c
static int find_pair(Pair *pairs, int count, const char *key) {
    for (int i = 0; i < count; i++) {
        if (strcmp(pairs[i].key, key) == 0) return i;
    }
    return -1;
}
```

### Decompiled

```c


int __cdecl find_pair(Pair *param_1,int param_2,char *param_3)



{

  int extraout_EAX;

  int local_24;



  local_24 = 0;

  while( true ) {

    if (param_2 <= local_24) {

      return -1;

    }

    strcmp();

    if (extraout_EAX == 0) break;

    local_24 = local_24 + 1;

  }

  return local_24;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for find_pair.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl find_pair(Pair *param_1,int param_2,char *param_3)



{

  int extraout_EAX;

  int local_24;



  local_24 = 0;

  while( true ) {

    if (param_2 <= local_24) {

      return -1;

    }

    strcmp();

    if (extraout_EAX == 0) break;

    local_24 = local_24 + 1;

  }

  return local_24;

}




```

### Qwen Via Prompt (base_qwen_openrouter)

- Summary: Raw Ghidra decompiler output without clarification.
- JSON valid: `False`
- Confidence: `0.000`

```c


int __cdecl find_pair(Pair *param_1,int param_2,char *param_3)



{

  int extraout_EAX;

  int local_24;



  local_24 = 0;

  while( true ) {

    if (param_2 <= local_24) {

      return -1;

    }

    strcmp();

    if (extraout_EAX == 0) break;

    local_24 = local_24 + 1;

  }

  return local_24;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Locate an existing key in the stored pair array.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int find_pair(Pair *pairs, int count, const char *key) {
    for (int i = 0; i < count; i++) {
        if (strcmp(pairs[i].key, key) == 0) return i;
    }
    return -1;
}
```

### GRPO Model

- Summary: Locate an existing key in the stored pair array.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int find_pair(Pair *pairs, int count, const char *key) {
    for (int i = 0; i < count; i++) {
        if (strcmp(pairs[i].key, key) == 0) return i;
    }
    return -1;
}
```

## fc3d59c077e1e126

- project_id: `cfg_token_router_001`
- function: `handle_command`
- task_type: `cleanup`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.492 | 0.000 |
| original_model | true | false | false | 0.492 | 0.000 |
| qwen_via_prompt | false | false | false | 0.492 | 0.000 |
| sft_model | true | false | false | 0.874 | 0.333 |
| grpo_model | true | false | false | 0.899 | 0.333 |

### Original Source

```c
static int handle_command(Store *st, char tokens[MAX_TOKENS][MAX_TOKEN_LEN], int ntok) {
    if (ntok == 0) return 1;
    if (strcmp(tokens[0], "set") == 0 && ntok == 3) {
        char *end = NULL;
        long v = strtol(tokens[2], &end, 10);
        if (end == tokens[2] || *end != '\0') return 0;
        return set_pair(st, tokens[1], (int)v);
    }
    if (strcmp(tokens[0], "add") == 0 && ntok == 3) {
        int cur = 0;
        char *end = NULL;
        long delta = strtol(tokens[2], &end, 10);
        if (end == tokens[2] || *end != '\0') return 0;
        if (!get_pair(st, tokens[1], &cur)) cur = 0;
        return set_pair(st, tokens[1], cur + (int)delta);
    }
    if (strcmp(tokens[0], "mul") == 0 && ntok == 3) {
        int cur = 0;
        char *end = NULL;
        long factor = strtol(tokens[2], &end, 10);
        if (end == tokens[2] || *end != '\0') return 0;
        if (!get_pair(st, tokens[1], &cur)) cur = 0;
        return set_pair(st, tokens[1], cur * (int)factor);
    }
    if (strcmp(tokens[0], "del") == 0 && ntok == 2) {
        int idx = find_pair(st, tokens[1]);
        if (idx < 0) return 1;
        for (size_t i = (size_t)idx + 1; i < st->count; i++) st->items[i - 1] = st->items[i];
        st->count--;
        return 1;
    }
    return 0;
}
```

### Decompiled

```c


int __cdecl handle_command(Store *param_1,char (*param_2) [32],int param_3)



{

  int extraout_EAX;

  int extraout_EAX_00;

  int iVar1;

  int extraout_EAX_01;

  long lVar2;

  int extraout_EAX_02;

  ulonglong local_50;

  char (*local_40) [32];

  int local_38;

  int local_34;

  char (*local_30) [32];

  int local_28;

  long local_24;

  char (*local_20) [32];

  Store *local_18;

  char (*local_10) [32];

  int local_8;

  int local_4;



  if (param_3 == 0) {

    local_4 = 1;

  }

  else {

    local_18 = param_1;

    local_10 = param_2;

    local_8 = param_3;

    strcmp();

    if ((extraout_EAX == 0) && (local_8 == 3)) {

      local_20 = (char (*) [32])0x0;

      local_24 = strtol(local_10[2],(char **)&local_20,10);

      if ((local_20 == local_10 + 2) || ((*local_20)[0] != '\0')) {

        local_4 = 0;

      }

      else {

        local_4 = set_pair(local_18,local_10[1],local_24);

      }

    }

    else {

      strcmp();

      if ((extraout_EAX_00 == 0) && (local_8 == 3)) {

        local_28 = 0;

        local_30 = (char (*) [32])0x0;

        local_34 = strtol(local_10[2],(char **)&local_30,10);

        if ((local_30 == local_10 + 2) || ((*local_30)[0] != '\0')) {

          local_4 = 0;

        }

        else {

          iVar1 = get_pair(local_18,local_10[1],&local_28);

          if (iVar1 == 0) {

            local_28 = 0;

          }

          local_4 = set_pair(local_18,local_10[1],local_28 + local_34);

        }

      }

      else {

        strcmp();

        if ((extraout_EAX_01 == 0) && (local_8 == 3)) {

          local_38 = 0;

          local_40 = (char (*) [32])0x0;

          lVar2 = strtol(local_10[2],(char **)&local_40,10);

          if ((local_40 == local_10 + 2) || ((*local_40)[0] != '\0')) {

            local_4 = 0;

          }

          else {

            iVar1 = get_pair(local_18,local_10[1],&local_38);

            if (iVar1 == 0) {

              local_38 = 0;

            }

            local_4 = set_pair(local_18,local_10[1],local_38 * lVar2);

          }

        }

        else {

          strcmp();

          if ((extraout_EAX_02 == 0) && (local_8 == 2)) {

            iVar1 = find_pair(local_18,local_10[1]);

            if (iVar1 < 0) {

              local_4 = 1;

            }

            else {

              local_50 = (ulonglong)iVar1;

              while (local_50 = local_50 + 1, local_50 < local_18->count) {

                memcpy();

              }

              local_18->count = local_18->count - 1;

              local_4 = 1;

            }

          }

          else {

            local_4 = 0;

          }

        }

      }

    }

  }

  return local_4;

}




```

### Original Model (prompt_only_cleanup)

- Summary: Prompt-free heuristic cleanup for handle_command.
- JSON valid: `True`
- Confidence: `0.350`

```c


int __cdecl handle_command(Store *param_1,char (*param_2) [32],int param_3)



{

  int extraout_EAX;

  int extraout_EAX_00;

  int iVar1;

  int extraout_EAX_01;

  long lVar2;

  int extraout_EAX_02;

  ulonglong local_50;

  char (*local_40) [32];

  int local_38;

  int local_34;

  char (*local_30) [32];

  int local_28;

  long local_24;

  char (*local_20) [32];

  Store *local_18;

  char (*local_10) [32];

  int local_8;

  int local_4;



  if (param_3 == 0) {

    local_4 = 1;

  }

  else {

    local_18 = param_1;

    local_10 = param_2;

    local_8 = param_3;

    strcmp();

    if ((extraout_EAX == 0) && (local_8 == 3)) {

      local_20 = (char (*) [32])0x0;

      local_24 = strtol(local_10[2],(char **)&local_20,10);

      if ((local_20 == local_10 + 2) || ((*local_20)[0] != '\0')) {

        local_4 = 0;

      }

      else {

        local_4 = set_pair(local_18,local_10[1],local_24);

      }

    }

    else {

      strcmp();

      if ((extraout_EAX_00 == 0) && (local_8 == 3)) {

        local_28 = 0;

        local_30 = (char (*) [32])0x0;

        local_34 = strtol(local_10[2],(char **)&local_30,10);

        if ((local_30 == local_10 + 2) || ((*local_30)[0] != '\0')) {

          local_4 = 0;

        }

        else {

          iVar1 = get_pair(local_18,local_10[1],&local_28);

          if (iVar1 == 0) {

            local_28 = 0;

          }

          local_4 = set_pair(local_18,local_10[1],local_28 + local_34);

        }

      }

      else {

        strcmp();

        if ((extraout_EAX_01 == 0) && (local_8 == 3)) {

          local_38 = 0;

          local_40 = (char (*) [32])0x0;

          lVar2 = strtol(local_10[2],(char **)&local_40,10);

          if ((local_40 == local_10 + 2) || ((*local_40)[0] != '\0')) {

            local_4 = 0;

          }

          else {

            iVar1 = get_pair(local_18,local_10[1],&local_38);

            if (iVar1 == 0) {

              local_38 = 0;

            }

            local_4 = set_pair(local_18,local_10[1],local_38 * lVar2);

          }

        }

        else {

          strcmp();

          if ((extraout_EAX_02 == 0) && (local_8 == 2)) {

            iVar1 = find_pair(local_18,local_10[1]);

            if (iVar1 < 0) {

              local_4 = 1;

            }

            else {

              local_50 = (ulonglong)iVar1;

              while (local_50 = local_50 + 1, local_50 < local_18->count) {

                memcpy();

              }

              local_18->count = local_18->count - 1;

              local_4 = 1;

            }

          }

          else {

            local_4 = 0;

          }

        }

      }

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


int __cdecl handle_command(Store *param_1,char (*param_2) [32],int param_3)



{

  int extraout_EAX;

  int extraout_EAX_00;

  int iVar1;

  int extraout_EAX_01;

  long lVar2;

  int extraout_EAX_02;

  ulonglong local_50;

  char (*local_40) [32];

  int local_38;

  int local_34;

  char (*local_30) [32];

  int local_28;

  long local_24;

  char (*local_20) [32];

  Store *local_18;

  char (*local_10) [32];

  int local_8;

  int local_4;



  if (param_3 == 0) {

    local_4 = 1;

  }

  else {

    local_18 = param_1;

    local_10 = param_2;

    local_8 = param_3;

    strcmp();

    if ((extraout_EAX == 0) && (local_8 == 3)) {

      local_20 = (char (*) [32])0x0;

      local_24 = strtol(local_10[2],(char **)&local_20,10);

      if ((local_20 == local_10 + 2) || ((*local_20)[0] != '\0')) {

        local_4 = 0;

      }

      else {

        local_4 = set_pair(local_18,local_10[1],local_24);

      }

    }

    else {

      strcmp();

      if ((extraout_EAX_00 == 0) && (local_8 == 3)) {

        local_28 = 0;

        local_30 = (char (*) [32])0x0;

        local_34 = strtol(local_10[2],(char **)&local_30,10);

        if ((local_30 == local_10 + 2) || ((*local_30)[0] != '\0')) {

          local_4 = 0;

        }

        else {

          iVar1 = get_pair(local_18,local_10[1],&local_28);

          if (iVar1 == 0) {

            local_28 = 0;

          }

          local_4 = set_pair(local_18,local_10[1],local_28 + local_34);

        }

      }

      else {

        strcmp();

        if ((extraout_EAX_01 == 0) && (local_8 == 3)) {

          local_38 = 0;

          local_40 = (char (*) [32])0x0;

          lVar2 = strtol(local_10[2],(char **)&local_40,10);

          if ((local_40 == local_10 + 2) || ((*local_40)[0] != '\0')) {

            local_4 = 0;

          }

          else {

            iVar1 = get_pair(local_18,local_10[1],&local_38);

            if (iVar1 == 0) {

              local_38 = 0;

            }

            local_4 = set_pair(local_18,local_10[1],local_38 * lVar2);

          }

        }

        else {

          strcmp();

          if ((extraout_EAX_02 == 0) && (local_8 == 2)) {

            iVar1 = find_pair(local_18,local_10[1]);

            if (iVar1 < 0) {

              local_4 = 1;

            }

            else {

              local_50 = (ulonglong)iVar1;

              while (local_50 = local_50 + 1, local_50 < local_18->count) {

                memcpy();

              }

              local_18->count = local_18->count - 1;

              local_4 = 1;

            }

          }

          else {

            local_4 = 0;

          }

        }

      }

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

- Summary: Validate and apply one supported command to the store.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int handle_command(const Store *st, const char *cmd, int argc) {
    if (argc == 0) return 1;
    if (strcmp(cmd, "set") == 0) {
        const char *key = cmd[2];
        const char *value = cmd[4];
        if (key[0] == '\0' || value[0] == '\0') return 0;
        return set_pair(st, key, (int64_t)strtoll(value, NULL, 10));
    }
    if (strcmp(cmd, "get") == 0) {
        const char *key = cmd[2];
        const int64_t *v = get_pair(st, key);
        if (!v) return 0;
        return *v;
    }
    if (strcmp(cmd, "find") == 0) {
        const char *key = cmd[2];
        const int64_t *v = find_pair(st, key);
        if (!v) return 1;
        return 1;
    }
    return 0;
}
```

### GRPO Model

- Summary: Validate and apply one supported command to the store.
- JSON valid: `True`
- Confidence: `1.000`

```c
static int handle_command(const Store *st, const char *cmd, int argc) {
    if (argc == 0) return 1;
    const char *key = cmd;
    const char *value;
    long v;
    if (strcmp(key, "set") == 0) {
        if (strcmp(cmd[1], "=") == 0) {
            v = strtol(cmd[2], NULL, 10);
            if (v == st->value && st->key) {
                st->value = v;
                return 1;
            }
        }
        return 0;
    }
    if (strcmp(key, "get") == 0) {
        if (strcmp(cmd[1], "=") == 0) {
            const int *it = find_pair(st, cmd[2]);
            if (it) {
                st->value = *it;
                return 1;
            }
        }
        return 0;
    }
    if (strcmp(key, "remove") == 0) {
        const int *it = find_pair(st, cmd[1]);
        if (it) {
            st->value = 0;
            return 1;
        }
        return 0;
    }
    return 0;
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

## d17a76948075b74f

- project_id: `c_token_config_parser`
- function: `main`
- task_type: `rename`
- compiler: `clang`
- opt_level: `O0`

| System | JSON | Compile | Behavior | Readability | Naming |
|:---|:---:|:---:|:---:|---:|---:|
| decompiled | true | false | false | 0.505 | 1.000 |
| original_model | true | false | false | 0.864 | 1.000 |
| qwen_via_prompt | false | false | false | 0.505 | 0.000 |
| sft_model | true | false | false | 0.866 | 1.000 |
| grpo_model | true | false | false | 0.868 | 1.000 |

### Original Source

```c
int main(void) {
    Pair items[MAX_ITEMS];
    int count = 0;
    char line[MAX_LINE];

    while (fgets(line, sizeof line, stdin)) {
        trim(line);
        if (line[0] == '\0') continue;
        if (count < MAX_ITEMS) {
            Pair p;
            if (split_pair(line, &p)) {
                items[count++] = p;
            }
        }
    }

    int mode_i = find_pair(items, count, "mode");
    int limit_i = find_pair(items, count, "limit");
    int flags_i = find_pair(items, count, "flags");

    uint16_t limit = 0;
    uint8_t flags = 0;
    int ok_mode = mode_i >= 0 && (strcmp(items[mode_i].value, "alpha") == 0 || strcmp(items[mode_i].value, "beta") == 0);
    int ok_limit = limit_i >= 0 && parse_u16(items[limit_i].value, &limit) && limit >= 10 && limit <= 99;
    int ok_flags = flags_i >= 0 && parse_hex8(items[flags_i].value, &flags);

    printf("items=%d\n", count);
    printf("mode_ok=%d\n", ok_mode);
    printf("limit_ok=%d\n", ok_limit);
    printf("flags_ok=%d\n", ok_flags);
    if (ok_limit) printf("limit=%u\n", (unsigned)limit);
    if (ok_flags) printf("flags=%u\n", (unsigned)flags);
    printf("ready=%d\n", (ok_mode && ok_limit && ok_flags) ? 1 : 0);
    return 0;
}
```

### Decompiled

```c


int __cdecl main(void)



{

  bool bVar1;

  int iVar2;

  int extraout_EAX;

  _iobuf *p_Var3;

  char *pcVar4;

  bool bVar5;

  uchar local_787;

  ushort local_786;

  int local_784;

  int local_780;

  int local_77c;

  Pair local_778;

  char local_718 [268];

  int local_60c;

  Pair local_608 [16];

  undefined4 local_4;



  local_4 = 0;

  local_60c = 0;

  while( true ) {

    p_Var3 = __acrt_iob_func(0);

    pcVar4 = fgets(local_718,0x100,p_Var3);

    if (pcVar4 == (char *)0x0) break;

    trim(local_718);

    if (((local_718[0] != '\0') && (local_60c < 0x10)) &&

       (iVar2 = split_pair(local_718,&local_778), iVar2 != 0)) {

      local_60c = local_60c + 1;

      memcpy();

    }

  }

  local_77c = find_pair(local_608,local_60c,"mode");

  local_780 = find_pair(local_608,local_60c,"limit");

  local_784 = find_pair(local_608,local_60c,"flags");

  local_786 = 0;

  local_787 = '\0';

  if ((-1 < local_77c) && (strcmp(), extraout_EAX != 0)) {

    strcmp();

  }

  bVar1 = false;

  if (-1 < local_780) {

    iVar2 = parse_u16(local_608[local_780].value,&local_786);

    bVar1 = false;

    if ((iVar2 != 0) && (bVar1 = false, 9 < local_786)) {

      bVar1 = local_786 < 100;

    }

  }

  bVar5 = false;

  if (-1 < local_784) {

    iVar2 = parse_hex8(local_608[local_784].value,&local_787);

    bVar5 = iVar2 != 0;

  }

  printf("items=%d\n");

  printf("mode_ok=%d\n");

  printf("limit_ok=%d\n");

  printf("flags_ok=%d\n");

  if (bVar1) {

    printf("limit=%u\n");

  }

  if (bVar5) {

    printf("flags=%u\n");

  }

  printf("ready=%d\n");

  return 0;

}




```

### Original Model (prompt_only_cleanup)

- Summary: The main function reads up to 16 lines from standard input, each expected to be a key-value pair string. It trims each line, splits it into a Pair structure, and stores it in an array. Then it searches this array for keys "mode", "limit", and "flags". It validates the "mode" key by comparing its value to a constant string. It parses "limit" as a 16-bit unsigned integer and checks if it is between 10 and 99 inclusive. It parses "flags" as an 8-bit hexadecimal value. Finally, it prints the count of items read, validation results for each key, the parsed limit and flags if valid, and a final ready status indicating if all validations passed.
- JSON valid: `True`
- Confidence: `1.000`

```c
int main(void) {
    int num_items = 0;
    char line_buffer[256];
    Pair pairs[16];
    Pair temp_pair;
    int mode_index, limit_index, flags_index;
    unsigned short limit_value = 0;
    unsigned char flags_value = 0;
    bool limit_ok = false;
    bool flags_ok = false;

    while (true) {
        FILE *stdin_file = __acrt_iob_func(0);
        char *fgets_result = fgets(line_buffer, 256, stdin_file);
        if (fgets_result == NULL) {
            break;
        }
        trim(line_buffer);
        if (line_buffer[0] != '\0' && num_items < 16) {
            int result = split_pair(line_buffer, &temp_pair);
            if (result != 0) {
                memcpy(&pairs[num_items], &temp_pair, sizeof(Pair));
                num_items++;
            }
        }
    }

    mode_index = find_pair(pairs, num_items, "mode");
    limit_index = find_pair(pairs, num_items, "limit");
    flags_index = find_pair(pairs, num_items, "flags");

    if (mode_index >= 0) {
        if (strcmp(pairs[mode_index].value, "expected_mode_string") == 0) {
            // mode is valid (details omitted)
        }
    }

    if (limit_index >= 0) {
        int result = parse_u16(pairs[limit_index].value, &limit_value);
        if (result != 0 && limit_value > 9 && limit_value < 100) {
            limit_ok = true;
        }
    }

    if (flags_index >= 0) {
        int result = parse_hex8(pairs[flags_index].value, &flags_value);
        flags_ok = (result != 0);
    }

    printf("items=%d\n", num_items);
    printf("mode_ok=%d\n", mode_index >= 0 /* && mode valid */);
    printf("limit_ok=%d\n", limit_ok);
    printf("flags_ok=%d\n", flags_ok);

    if (limit_ok) {
        printf("limit=%u\n", limit_value);
    }
    if (flags_ok) {
        printf("flags=%u\n", flags_value);
    }

    printf("ready=%d\n", (mode_index >= 0 /* && mode valid */) && limit_ok && flags_ok);

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

  bool bVar1;

  int iVar2;

  int extraout_EAX;

  _iobuf *p_Var3;

  char *pcVar4;

  bool bVar5;

  uchar local_787;

  ushort local_786;

  int local_784;

  int local_780;

  int local_77c;

  Pair local_778;

  char local_718 [268];

  int local_60c;

  Pair local_608 [16];

  undefined4 local_4;



  local_4 = 0;

  local_60c = 0;

  while( true ) {

    p_Var3 = __acrt_iob_func(0);

    pcVar4 = fgets(local_718,0x100,p_Var3);

    if (pcVar4 == (char *)0x0) break;

    trim(local_718);

    if (((local_718[0] != '\0') && (local_60c < 0x10)) &&

       (iVar2 = split_pair(local_718,&local_778), iVar2 != 0)) {

      local_60c = local_60c + 1;

      memcpy();

    }

  }

  local_77c = find_pair(local_608,local_60c,"mode");

  local_780 = find_pair(local_608,local_60c,"limit");

  local_784 = find_pair(local_608,local_60c,"flags");

  local_786 = 0;

  local_787 = '\0';

  if ((-1 < local_77c) && (strcmp(), extraout_EAX != 0)) {

    strcmp();

  }

  bVar1 = false;

  if (-1 < local_780) {

    iVar2 = parse_u16(local_608[local_780].value,&local_786);

    bVar1 = false;

    if ((iVar2 != 0) && (bVar1 = false, 9 < local_786)) {

      bVar1 = local_786 < 100;

    }

  }

  bVar5 = false;

  if (-1 < local_784) {

    iVar2 = parse_hex8(local_608[local_784].value,&local_787);

    bVar5 = iVar2 != 0;

  }

  printf("items=%d\n");

  printf("mode_ok=%d\n");

  printf("limit_ok=%d\n");

  printf("flags_ok=%d\n");

  if (bVar1) {

    printf("limit=%u\n");

  }

  if (bVar5) {

    printf("flags=%u\n");

  }

  printf("ready=%d\n");

  return 0;

}




```

Raw model text:
```text
OPENROUTER_ERROR: {"error":{"message":"Qwen/Qwen3.5-2B is not a valid model ID","code":400},"user_id":"<redacted>"}
```

### SFT Model

- Summary: Parse a tiny configuration format from stdin, validate a few typed fields, and report normalized status lines.
- JSON valid: `True`
- Confidence: `1.000`

```c
int main(void) {
    char line[MAX_LINE];
    Pair pairs[MAX_ITEMS];
    int count = 0;
    int mode_ok = 0;
    int limit_ok = 0;
    int flags_ok = 0;
    int valid = 0;
    int limit = 0;
    char *mode = NULL;
    char *limit = NULL;
    char *flags = NULL;
    char *trimmed;

    while (fgets(line, sizeof(line), stdin)) {
        trimmed = trim(line);
        if (trimmed[0] == '\0' || count >= MAX_ITEMS) continue;
        if (split_pair(trimmed, &pairs[count])) {
            count++;
            if (strcmp(pairs[count].key, "mode") == 0) mode_ok = 1;
            if (strcmp(pairs[count].key, "limit") == 0) limit_ok = 1;
            if (strcmp(pairs[count].key, "flags") == 0) flags_ok = 1;
        }
    }

    if (mode_ok && limit_ok && flags_ok) {
        int ok = 0;
        if (limit > 0 && limit < 100) ok = 1;
        if (parse_u16(pairs[count].value, &limit) && ok) ok = 1;
        if (parse_hex8(pairs[count].value, &flags) && ok) ok = 1;
        printf("items=%d\n", count);
        printf("mode_ok=%d\n", mode_ok);
        printf("limit_ok=%d\n", limit_ok);
        printf("flags_ok=%d\n", flags_ok);
        if (ok) printf("limit=%u\n", limit);
        if (ok) printf("flags=%u\n", (unsigned)flags);
        printf("ready=%d\n", ok);
        return 0;
    }
    return 1;
}
```

### GRPO Model

- Summary: Parse a tiny configuration format from stdin, validate a few typed fields, and report normalized status lines.
- JSON valid: `True`
- Confidence: `1.000`

```c
int main(void) {
    char line[MAX_LINE];
    Pair pairs[MAX_ITEMS];
    int count = 0;
    int mode_ok = 0;
    int limit_ok = 0;
    int flags_ok = 0;
    int valid = 0;
    int limit = 0;
    char flags[16];
    char mode[MAX_ITEMS];
    char limit[MAX_ITEMS];

    while (fgets(line, sizeof(line), stdin)) {
        trim(line);
        if (line[0] == '\0' || count >= MAX_ITEMS) continue;
        if (split_pair(line, &pairs[count])) {
            count++;
            if (strcmp(pairs[count].key, "mode") == 0) mode_ok = 1;
            if (strcmp(pairs[count].key, "limit") == 0) {
                if (parse_u16(pairs[count].value, &limit)) {
                    if (limit >= 100) mode_ok = 0;
                }
            }
            if (strcmp(pairs[count].key, "flags") == 0) {
                if (parse_hex8(pairs[count].value, flags)) {
                    flags_ok = 1;
                }
            }
        }
    }

    if (mode_ok) printf("mode_ok=%d\n", mode_ok);
    if (limit_ok) printf("limit_ok=%d\n", limit_ok);
    if (flags_ok) printf("flags_ok=%d\n", flags_ok);
    if (valid) printf("limit=%u\n", limit);
    if (flags_ok) printf("flags=%u\n", (unsigned)strlen(flags));
    printf("ready=%d\n", valid);
    return 0;
}
```
