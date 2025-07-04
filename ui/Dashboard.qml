import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtCharts 2.15

ApplicationWindow {
    id: dashboard
    width: 900; height: 650
    visible: true
    title: "Stress Dashboard"
    color: "white"

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12
            Layout.margins: 16

            // ◀ Left stats panel
            Frame {
                Layout.preferredWidth: 160
                Layout.fillHeight: true
                Column {
                    anchors.centerIn: parent
                    spacing: 12
                    Text {
                        text: "Avg Stress"
                        font.pointSize: 14
                        horizontalAlignment: Text.AlignHCenter
                    }
                    Text {
                        font.pointSize: 24
                        horizontalAlignment: Text.AlignHCenter
                        text: {
                            var sum = 0
                            for (var i = 0; i < stressModel.rowCount(); ++i)
                                sum += stressModel.get(i).score
                            return (sum / stressModel.rowCount()).toFixed(1) + "%"
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

                    // use BarCategoryAxis (which DOES have 'categories')
                    BarCategoryAxis {
                        id: xAxis
                        categories: stressModel.date
                    }
                    ValueAxis {
                        id: yAxis
                        min: 0; max: 100
                    }

                    BarSeries {
                        axisX: xAxis
                        axisY: yAxis
                        BarSet {
                            label: "Stress"
                            values: stressModel.score
                        }
                    }
                }

                Rectangle {
                    id: startBtn
                    width: 130; height: 130; radius: width/2; color: "red"
                    anchors.centerIn: parent
                    MouseArea { anchors.fill: parent; onClicked: controller.startWorkSession() }
                }

                Text {
                    text: "Start Today's\nWork Session"
                    font.pixelSize: 14
                    horizontalAlignment: Text.AlignHCenter
                    anchors.horizontalCenter: startBtn.horizontalCenter
                    anchors.top: startBtn.bottom; anchors.topMargin: 8
                }
            }

            // ▶ Right stats panel
            Frame {
                Layout.preferredWidth: 160
                Layout.fillHeight: true
                Column {
                    anchors.centerIn: parent
                    spacing: 12
                    Text {
                        text: "Max Stress"
                        font.pointSize: 14
                        horizontalAlignment: Text.AlignHCenter
                    }
                    Text {
                        font.pointSize: 24
                        horizontalAlignment: Text.AlignHCenter
                        text: {
                            var m = 0
                            for (var i = 0; i < stressModel.rowCount(); ++i)
                                m = Math.max(m, stressModel.get(i).score)
                            return m + "%"
                        }
                    }
                }
            }
        }

        // ─── Bottom bar ───────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            height: 60
            color: "#efefef"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 24
                Button { text: "History" }
                Button { text: "Settings" }
                Button { text: "Help" }
            }
        }
    }
}
