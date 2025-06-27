import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true               
    width: 400; height: 300
    title: "Mood Mirror"

    Column {
        anchors.centerIn: parent
        spacing: 20

        Text { id: statusText; text: "Initializing..." }
        Button {
            text: "Force Check"
            onClicked: controller.forceCheck()
        }
    }
}
