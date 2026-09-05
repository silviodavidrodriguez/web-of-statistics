(function () {
    "use strict";

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function isMarkdownSeparatorRow(row) {
        return row.length > 0
            && row.every(function (value) {
                return /^:?-{2,}:?$/.test(value.trim());
            });
    }

    function splitPreviewRow(line) {
        var trimmed = line.trim();

        if (trimmed.indexOf("\t") !== -1) {
            return trimmed.split("\t").map(function (value) {
                return value.trim();
            });
        }

        if (trimmed.indexOf("|") !== -1) {
            trimmed = trimmed.replace(/^\|/, "").replace(/\|$/, "");
            return trimmed.split("|").map(function (value) {
                return value.trim();
            });
        }

        if (trimmed.indexOf(";") !== -1) {
            return trimmed.split(";").map(function (value) {
                return value.trim();
            });
        }

        return trimmed.split(/\s+/).map(function (value) {
            return value.trim();
        });
    }

    function parsePreviewRows(rawText) {
        return rawText
            .replace(/\r/g, "")
            .replace(/\n+$/g, "")
            .split("\n")
            .filter(function (line) {
                return line.trim() !== "";
            })
            .map(splitPreviewRow)
            .filter(function (row) {
                return row.length > 0 && !isMarkdownSeparatorRow(row);
            });
    }

    function createPreview(textareaId) {
        var textarea = document.getElementById(textareaId);
        if (!textarea) return;

        var table = document.querySelector('[data-preview-for="' + textareaId + '"]');
        var wrapper = document.querySelector('[data-preview-wrapper-for="' + textareaId + '"]');
        if (!table || !wrapper) return;

        function renderPreview() {
            var text = textarea.value;

            if (text.trim() === "") {
                table.innerHTML = "";
                wrapper.classList.remove("has-data");
                textarea.classList.remove("has-preview");
                return;
            }

            var rows = parsePreviewRows(text);

            if (!rows.length) {
                table.innerHTML = "";
                wrapper.classList.remove("has-data");
                textarea.classList.remove("has-preview");
                return;
            }

            var html = "<tbody>";

            rows.forEach(function (row) {
                html += "<tr>";

                row.forEach(function (cell) {
                    var value = escapeHtml(cell);
                    if (value === "") value = "&nbsp;";
                    html += "<td>" + value + "</td>";
                });

                html += "</tr>";
            });

            html += "</tbody>";

            table.innerHTML = html;
            wrapper.classList.add("has-data");
            textarea.classList.add("has-preview");
        }

        textarea.addEventListener("input", renderPreview);
        textarea.addEventListener("paste", function () {
            window.setTimeout(renderPreview, 50);
        });

        renderPreview();
    }

    function updateCapabilitySigmaField() {
        var selector = document.getElementById("sigma_method");
        var field = document.getElementById("control-within-sigma-field");
        if (!selector || !field) return;

        field.classList.toggle("is-hidden", selector.value !== "provided");
    }

    function updateVMaskNote() {
        var selector = document.getElementById("cusum_method");
        var note = document.getElementById("control-vmask-note");
        if (!selector || !note) return;

        note.classList.toggle("is-hidden", selector.value !== "vmask");
    }

    document.addEventListener("DOMContentLoaded", function () {
        createPreview("data");

        var sigmaSelector = document.getElementById("sigma_method");
        if (sigmaSelector) {
            sigmaSelector.addEventListener("change", updateCapabilitySigmaField);
            updateCapabilitySigmaField();
        }

        var cusumSelector = document.getElementById("cusum_method");
        if (cusumSelector) {
            cusumSelector.addEventListener("change", updateVMaskNote);
            updateVMaskNote();
        }
    });
}());
