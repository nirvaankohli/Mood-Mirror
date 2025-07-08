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
                    model: ["Last 7 days", "Last 30 days", "Last 3 months", "Last 6 months", "All time"]
                    currentIndex: 0
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                    background: Rectangle { color: "#4a4a4a"; radius: 8 }
                    contentItem: Text { text: timeRangeCombo.currentText; color: "#ffffff"; font.pixelSize: 14 }
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
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12
            Layout.margins: 16

            // ◀ Left stats panel
            Frame {
                Layout.preferredWidth: 160
                Layout.fillHeight: true
                background: Rectangle { color: "#3a3a3a"; radius: 12 }
                Column {
                    anchors.centerIn: parent
                    spacing: 8
                    Text { text: "Avg Stress"; font.pointSize: 14; color: "#ffffff"; horizontalAlignment: Text.AlignHCenter }
                    Text {
                        font.pointSize: 24
                        color: "#ffffff"
                        horizontalAlignment: Text.AlignHCenter
                        text: {
                            var sum = 0; for (var i = 0; i < stressModel.rowCount(); ++i) sum += stressModel.get(i).score;
                            return (sum / stressModel.rowCount()).toFixed(1) + "%";
                        }
                    }
                }
            }

            // ▼ Center: Chart + Red Button
            Item {
                Layout.fillWidth: true
                Layout.fillHeight: true

                ChartView {
                    anchors.fill: parent
                    antialiasing: true
                    // Chart background
                    backgroundColor: "#1e1e1e"

                    BarCategoryAxis { id: xAxis; categories: stressModel.date; labelsColor: "#ffffff" }
                    ValueAxis { id: yAxis; min: 0; max: 100; labelsColor: "#ffffff" }
                    BarSeries { axisX: xAxis; axisY: yAxis; BarSet { label: "Stress"; values: stressModel.score } }
                }

                // Contrasting ring
                Rectangle {
                    id: ring
                    width: 160; height: 160; radius: 80
                    color: "#262626"
                    anchors.centerIn: parent
                }

                // Record button
                Rectangle {
                    id: startBtn
                    width: 130; height: 130; radius: 65
                    color: "#e74c3c"
                    anchors.centerIn: parent
                    MouseArea { anchors.fill: parent; onClicked: controller.startWorkSession() }
                }

                Text {
                    text: "Start Today's\nWork Session"
                    font.pixelSize: 14; color: "#ffffff"
                    horizontalAlignment: Text.AlignHCenter
                    anchors.horizontalCenter: startBtn.horizontalCenter
                    anchors.top: startBtn.bottom; anchors.topMargin: 8
                }
            }

            // ▶ Right stats panel
            Frame {
                Layout.preferredWidth: 160
                Layout.fillHeight: true
                background: Rectangle { color: "#3a3a3a"; radius: 12 }
                Column {
                    anchors.centerIn: parent
                    spacing: 8
                    Text { text: "Max Stress"; font.pointSize: 14; color: "#ffffff"; horizontalAlignment: Text.AlignHCenter }
                    Text {
                        font.pointSize: 24; color: "#ffffff"; horizontalAlignment: Text.AlignHCenter
                        text: {
                            var m = 0; for (var i = 0; i < stressModel.rowCount(); ++i) m = Math.max(m, stressModel.get(i).score);
                            return m + "%";
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
            }
        }
    }
}
