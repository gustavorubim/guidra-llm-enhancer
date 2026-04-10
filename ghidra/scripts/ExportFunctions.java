import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlockIterator;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Variable;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ExportFunctions extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length == 0) {
            throw new IllegalArgumentException("ExportFunctions.java requires an output directory argument");
        }

        File outputDir = new File(args[0]);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("Failed to create output directory: " + outputDir.getAbsolutePath());
        }

        File functionsFile = new File(outputDir, "functions.jsonl");
        File manifestFile = new File(outputDir, "project_manifest.json");
        File decompiledSnapshotFile = new File(outputDir, "decompiled.c");
        File disassemblySnapshotFile = new File(outputDir, "disassembly.txt");
        File stringsFile = new File(outputDir, "strings.txt");
        File importsFile = new File(outputDir, "imports.txt");
        File symbolsFile = new File(outputDir, "symbols.txt");

        DecompInterface decompiler = new DecompInterface();
        DecompileOptions options = new DecompileOptions();
        decompiler.setOptions(options);
        decompiler.openProgram(currentProgram);

        FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
        List<String> decompiledSections = new ArrayList<>();
        List<String> disassemblySections = new ArrayList<>();
        List<String> importNames = new ArrayList<>();
        List<String> symbolNames = new ArrayList<>();

        try (BufferedWriter functionWriter = new BufferedWriter(new FileWriter(functionsFile))) {
            while (functions.hasNext() && !monitor.isCancelled()) {
                Function function = functions.next();
                symbolNames.add(function.getName());

                Map<String, Object> record = exportFunction(decompiler, function);
                functionWriter.write(toJson(record));
                functionWriter.newLine();

                decompiledSections.add("// " + function.getName() + "\n" + (String) record.get("decompiled_text"));
                disassemblySections.add("; " + function.getName() + "\n" + (String) record.get("disassembly_text"));

                @SuppressWarnings("unchecked")
                List<String> imports = (List<String>) record.get("imports");
                importNames.addAll(imports);
            }
        }

        writeJoinedFile(decompiledSnapshotFile, decompiledSections, "\n\n");
        writeJoinedFile(disassemblySnapshotFile, disassemblySections, "\n\n");
        writeJoinedFile(stringsFile, collectDefinedStrings(), "\n");
        writeJoinedFile(importsFile, importNames, "\n");
        writeJoinedFile(symbolsFile, symbolNames, "\n");

        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("project_id", currentProgram.getName());
        manifest.put("binary_path", currentProgram.getExecutablePath());
        manifest.put("binary_name", currentProgram.getName());
        manifest.put("output_dir", outputDir.getAbsolutePath());
        manifest.put("functions_path", functionsFile.getAbsolutePath());
        manifest.put("strings_path", stringsFile.getAbsolutePath());
        manifest.put("imports_path", importsFile.getAbsolutePath());
        manifest.put("symbols_path", symbolsFile.getAbsolutePath());
        manifest.put("decompiled_snapshot_path", decompiledSnapshotFile.getAbsolutePath());
        manifest.put("disassembly_snapshot_path", disassemblySnapshotFile.getAbsolutePath());
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(manifestFile))) {
            writer.write(toJson(manifest));
            writer.newLine();
        }
    }

    private Map<String, Object> exportFunction(DecompInterface decompiler, Function function) throws Exception {
        Map<String, Object> record = new LinkedHashMap<>();
        DecompileResults results = decompiler.decompileFunction(function, 60, monitor);
        String decompiledText = "";
        if (results != null && results.getDecompiledFunction() != null) {
            decompiledText = results.getDecompiledFunction().getC();
        }

        record.put("project_id", currentProgram.getName());
        record.put("binary_path", currentProgram.getExecutablePath());
        record.put("binary_name", currentProgram.getName());
        record.put("function_address", function.getEntryPoint().toString());
        record.put("ghidra_function_name", function.getName());
        record.put("signature", function.getSignature().getPrototypeString());
        record.put("return_type", function.getReturnType().getDisplayName());
        record.put("parameters", exportParameters(function));
        record.put("local_variables", exportLocals(function));
        record.put("decompiled_text", decompiledText);
        record.put("disassembly_text", exportDisassembly(function));
        record.put("strings", new ArrayList<String>());
        record.put("imports", exportImports(function));
        record.put("callees", exportCallees(function));
        record.put("callers", exportCallers(function));
        record.put("basic_block_count", countBasicBlocks(function));
        record.put("instruction_count", countInstructions(function));
        return record;
    }

    private List<Map<String, String>> exportParameters(Function function) {
        List<Map<String, String>> params = new ArrayList<>();
        for (Parameter parameter : function.getParameters()) {
            Map<String, String> record = new LinkedHashMap<>();
            record.put("name", parameter.getName());
            record.put("type", parameter.getDataType().getDisplayName());
            record.put("storage", parameter.getVariableStorage().toString());
            params.add(record);
        }
        return params;
    }

    private List<Map<String, String>> exportLocals(Function function) {
        List<Map<String, String>> locals = new ArrayList<>();
        try {
            for (Variable local : function.getLocalVariables()) {
                Map<String, String> record = new LinkedHashMap<>();
                record.put("name", local.getName());
                record.put("type", local.getDataType().getDisplayName());
                record.put("storage", local.getVariableStorage().toString());
                locals.add(record);
            }
        } catch (Exception ignored) {
        }
        return locals;
    }

    private String exportDisassembly(Function function) {
        StringBuilder builder = new StringBuilder();
        Listing listing = currentProgram.getListing();
        for (Instruction instruction : listing.getInstructions(function.getBody(), true)) {
            builder.append(instruction.getAddress().toString())
                .append("  ")
                .append(instruction.toString())
                .append("\n");
        }
        return builder.toString().trim();
    }

    private List<String> exportImports(Function function) throws Exception {
        List<String> imports = new ArrayList<>();
        for (Function callee : function.getCalledFunctions(monitor)) {
            if (callee.isExternal()) {
                imports.add(callee.getName());
            }
        }
        return imports;
    }

    private List<String> exportCallees(Function function) throws Exception {
        List<String> callees = new ArrayList<>();
        for (Function callee : function.getCalledFunctions(monitor)) {
            callees.add(callee.getName());
        }
        return callees;
    }

    private List<String> exportCallers(Function function) throws Exception {
        List<String> callers = new ArrayList<>();
        for (Function caller : function.getCallingFunctions(monitor)) {
            callers.add(caller.getName());
        }
        return callers;
    }

    private int countBasicBlocks(Function function) {
        try {
            BasicBlockModel model = new BasicBlockModel(currentProgram);
            CodeBlockIterator iterator = model.getCodeBlocksContaining(function.getBody(), monitor);
            int count = 0;
            while (iterator.hasNext()) {
                iterator.next();
                count += 1;
            }
            return count;
        } catch (Exception exc) {
            return 0;
        }
    }

    private int countInstructions(Function function) {
        int count = 0;
        for (Instruction ignored : currentProgram.getListing().getInstructions(function.getBody(), true)) {
            count += 1;
        }
        return count;
    }

    private List<String> collectDefinedStrings() {
        List<String> strings = new ArrayList<>();
        Listing listing = currentProgram.getListing();
        listing.getDefinedData(true).forEachRemaining(data -> {
            try {
                Object value = data.getValue();
                if (value instanceof String) {
                    strings.add((String) value);
                }
            } catch (Exception ignored) {
            }
        });
        return strings;
    }

    private void writeJoinedFile(File file, List<String> lines, String separator) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            boolean first = true;
            for (String line : lines) {
                if (!first) {
                    writer.write(separator);
                }
                writer.write(line);
                first = false;
            }
        }
    }

    private String toJson(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof String) {
            return "\"" + escapeJson((String) value) + "\"";
        }
        if (value instanceof Number || value instanceof Boolean) {
            return value.toString();
        }
        if (value instanceof Map) {
            StringBuilder builder = new StringBuilder();
            builder.append("{");
            boolean first = true;
            for (Object entryObject : ((Map<?, ?>) value).entrySet()) {
                Map.Entry<?, ?> entry = (Map.Entry<?, ?>) entryObject;
                if (!first) {
                    builder.append(",");
                }
                builder.append(toJson(entry.getKey().toString()));
                builder.append(":");
                builder.append(toJson(entry.getValue()));
                first = false;
            }
            builder.append("}");
            return builder.toString();
        }
        if (value instanceof Iterable) {
            StringBuilder builder = new StringBuilder();
            builder.append("[");
            boolean first = true;
            for (Object item : (Iterable<?>) value) {
                if (!first) {
                    builder.append(",");
                }
                builder.append(toJson(item));
                first = false;
            }
            builder.append("]");
            return builder.toString();
        }
        return toJson(value.toString());
    }

    private String escapeJson(String value) {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t");
    }
}
