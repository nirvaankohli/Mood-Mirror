import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtCharts 2.15

ApplicationWindow {
    id: dashboard
    width: 900; height: 650
    visible: true
    title: "Stress Dashboard"
    color: "#2f2f2f"

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        // Top visual banner with dropdowns
        Rectangle {
            Layout.fillWidth: true
            height: 120
            color: "#353535"
            radius: 16

            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                // Time range selector
                ComboBox {
                    id: timeRangeCombo
                    model: ["Today", "Last 7 days", "Last 30 days", "Last 3 months", "Last 6 months", "All time"]
                    currentIndex: 0
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                    background: Rectangle { color: "#4a4a4a"; radius: 8 }
                    contentItem: Text { text: timeRangeCombo.currentText; color: "#ffffff"; font.pixelSize: 14 }
                    onCurrentIndexChanged: {
                        if (stressModel) stressModel.reload(timeRangeCombo.currentIndex, metricCombo.currentText)
                    }
                }

                // Spacer to center the title
                Item { Layout.fillWidth: true }

                // Title
                Text {
                    text: "Stress Monitoring"
                    font.pixelSize: 20
                    color: "#ffffff"
                    font.bold: true
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
                }

                // Spacer to push the right dropdown
                Item { Layout.fillWidth: true }

                // Metric selector
                ComboBox {
                    id: metricCombo
                    model: ["Stress score", "Total interventions"]
                    currentIndex: 0
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                    background: Rectangle { color: "#4a4a4a"; radius: 8 }
                    contentItem: Text { text: metricCombo.currentText; color: "#ffffff"; font.pixelSize: 14 }
                    onCurrentIndexChanged: {
                        if (stressModel) stressModel.reload(timeRangeCombo.currentIndex, metricCombo.currentText)
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 24
            Layout.margins: 24

            Frame {

                Layout.preferredWidth: 180
                
                Layout.fillHeight: true

                background: Rectangle { color: "#23272e"; radius: 18 }

                Column {

                    anchors.centerIn: parent
                    spacing: 12

                    Text { text: "Avg Stress"; font.pointSize: 16; color: "#b0b8c1"; horizontalAlignment: Text.AlignHCenter }
                    
                    Text {

                        font.pointSize: 32
                        color: "#00b894"
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter

                        text: {
                            if (!stressModel) return "0.0";
                            var sum = 0; for (var i = 0; i < stressModel.rowCount(); ++i) sum += stressModel.get(i).score;
                            return (sum / Math.max(1, stressModel.rowCount())).toFixed(1);
                        }
                    }
                }
            }

            // ▼ Center: Chart + Red Button
            Item {

                Layout.fillWidth: true
                Layout.fillHeight: true

                Rectangle {

                    anchors.fill: parent

                    color: "#23272e"

                    radius: 24

                    border.color: "#353b48"
                    border.width: 2

                    anchors.margins: 8

                    ChartView {

                        anchors.fill: parent
                        anchors.margins: 24

                        antialiasing: true

                        backgroundColor: "#23272e"
                        plotAreaColor: "#23272e"

                        legend.visible: false
                        theme: ChartView.ChartThemeDark

                        LineSeries {

                            id: lineSeries

                            name: "Stress"

                            color: "#00b894"

                            width: 3

                            useOpenGL: true
                            
                            axisX: xAxis
                            axisY: yAxis

                        }
                        
                        ScatterSeries {

                            id: scatterSeries
                            name: "Data Points"

                            color: "#00b894"

                            markerSize: 6

                            borderColor: "#ffffff"

                            borderWidth: 1
                            
                            axisX: xAxis
                            axisY: yAxis

                        }



                        ValueAxis {

                            id: xAxis

                            min: 0
                            max: Math.max(1, stressModel.rowCount() - 1)

                            tickCount: Math.min(8, stressModel.rowCount())

                            labelsColor: "#b0b8c1"
                            gridLineColor: "#353b48"

                            gridVisible: true
                            minorTickCount: 1
                            
                            labelFormat: "%.0f"

                        }
                        

                        ValueAxis {

                            id: yAxis

                            min: 0
                            max: getMaxStressValue()

                            labelsColor: "#b0b8c1"
                            gridLineColor: "#353b48"

                            gridVisible: true

                            minorTickCount: 1

                            labelFormat: "%.0f"

                        }

                        function getMaxStressValue() {

                            if (!stressModel || stressModel.rowCount() === 0) {

                                console.log("DEBUG: getMaxStressValue - no data, returning 10");
                                return 10;

                            }

                            var max = 0;

                            for (var i = 0; i < stressModel.rowCount(); ++i) {

                                var score = stressModel.get(i).score;
                                max = Math.max(max, score);

                                console.log("DEBUG: getMaxStressValue - checking score:", score, "max so far:", max);
                            
                            }

                            var result = Math.max(10, Math.ceil(max * 1.2));

                            console.log("DEBUG: getMaxStressValue - final result:", result);
                            
                            return result;

                        }

                        function updateLineData() {

                            if (!stressModel) return;
                            
                            console.log("DEBUG: updateLineData called, stressModel.rowCount():", stressModel.rowCount());
                            

                            lineSeries.clear();
                            scatterSeries.clear();
                            

                            var categories = [];

                            for (var i = 0; i < stressModel.rowCount(); ++i) {

                                var entry = stressModel.get(i);

                                console.log("DEBUG: Entry", i, "- date:", entry.date, "score:", entry.score);
                                categories.push(entry.date);
                                
                                lineSeries.append(i, entry.score);
                                scatterSeries.append(i, entry.score);

                            }
                            
                            xAxis.categories = categories;
                            
                            // Update y-axis max
                            yAxis.max = getMaxStressValue();
                            
                            console.log("DEBUG: Line series points:", lineSeries.count);
                            console.log("DEBUG: Scatter series points:", scatterSeries.count);
                        }



                        Connections {
                            target: stressModel
                            function onModelReset() { 
                                updateLineData(); 
                            }
                        }
                        
                        Component.onCompleted: updateLineData()
                    }
                    // Contrasting ring
                    Rectangle {
                        id: ring
                        width: 160; height: 160; radius: 80
                        color: "#1e2329"
                        anchors.centerIn: parent
                        border.color: "#00b894"
                        border.width: 3
                    }
                    // Record button
                    Rectangle {
                        id: startBtn
                        width: 130; height: 130; radius: 65
                        color: "#e74c3c"
                        anchors.centerIn: parent
                        border.color: "#fff"
                        border.width: 2
                        MouseArea { anchors.fill: parent; onClicked: controller.startWorkSession() }
                        Text {
                            text: "Start\nSession"
                            font.pixelSize: 16; color: "#fff"
                            anchors.centerIn: parent
                            horizontalAlignment: Text.AlignHCenter
                        }
                    }
                    
                    // Button label
                    Text {
                        text: "Start Today's\nWork Session"
                        font.pixelSize: 14; color: "#b0b8c1"
                        horizontalAlignment: Text.AlignHCenter
                        anchors.horizontalCenter: startBtn.horizontalCenter
                        anchors.top: startBtn.bottom; anchors.topMargin: 8
                    }
                }
            }

            // ▶ Right stats panel
            Frame {
                Layout.preferredWidth: 180
                Layout.fillHeight: true
                background: Rectangle { color: "#23272e"; radius: 18 }
                Column {
                    anchors.centerIn: parent
                    spacing: 12
                    Text { text: "Max Stress"; font.pointSize: 16; color: "#b0b8c1"; horizontalAlignment: Text.AlignHCenter }
                    Text {
                        font.pointSize: 32; color: "#e17055"; font.bold: true; horizontalAlignment: Text.AlignHCenter
                        text: {
                            if (!stressModel) return "0";
                            var maxSession = stressModel.get_max_stress_session();
                            if (maxSession && maxSession.stress_score) {
                                return Math.round(maxSession.stress_score);
                            } else {
                                var m = 0; for (var i = 0; i < stressModel.rowCount(); ++i) m = Math.max(m, stressModel.get(i).score);
                                return Math.round(m);
                            }
                        }
                    }
                }
            }
        }

        // ─── Bottom bar ───────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            height: 60
            color: "#353535"
            radius: 12
            anchors.margins: 16

            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 24
                Button { text: "History"; background: Rectangle { color: "#3a3a3a"; radius: 8 } }
                Button { text: "Settings"; background: Rectangle { color: "#3a3a3a"; radius: 8 } }
                Button { text: "Help"; background: Rectangle { color: "#3a3a3a"; radius: 8 } }
                Button {
                    text: "Debug Model"
                    onClicked: {
                        if (!stressModel) {
                            console.log("stressModel is null");
                            return;
                        }
                        console.log("Row count:", stressModel.rowCount());
                        for (var i = 0; i < stressModel.rowCount(); ++i)
                            console.log("Entry", i, "date:", stressModel.get(i).date, "score:", stressModel.get(i).score);
                        var maxSession = stressModel.get_max_stress_session();
                        console.log("Max stress session:", maxSession);
                    }
                }
            }
        }
    }
}
