import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    id: window
    visible: true
    width: 400; height: 300
    title: "Mood Mirror"

    FontLoader { id: satoshi; source: Qt.resolvedUrl("../assets/Satoshi-Variable.ttf") }
    FontLoader { id: satoshiItalic; source: Qt.resolvedUrl("../assets/Satoshi-VariableItalic.ttf") }

    property string userName: backend.userName
    property string tempName: ""

    Component.onCompleted: {
        tempName = userName;
        nameField.text = tempName;
    }

    Image {
        anchors.fill: parent
        source: Qt.resolvedUrl("../assets/calm_scenery.png")
        fillMode: Image.PreserveAspectCrop
        z: -1
    }
    Rectangle {
        anchors.fill: parent
        color: "#80000000"
        z: -1
    }

    Column {
        anchors.centerIn: parent
        spacing: 20
        width: parent.width * 0.8
        anchors.horizontalCenter: parent.horizontalCenter

        Text {
            text: "Welcome"
            font.family: satoshi.name
            font.pixelSize: window.height * 0.15 + 10
            font.weight: Font.Normal
            color: "white"
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
        }

        TextField {
            id: nameField
            width: parent.width
            text: tempName
            placeholderText: "Enter your name"
            placeholderTextColor: "#DDD"
            color: "white"
            font.family: satoshiItalic.name
            font.pixelSize: window.height * 0.1
            font.weight: Font.Normal
            horizontalAlignment: Text.AlignHCenter
            onTextChanged: tempName = text
            background: Rectangle {
                color: "transparent"
                border.width: 0
                Rectangle {
                    anchors {
                        left: parent.left; right: parent.right; bottom: parent.bottom
                    }
                    height: 2
                    color: (nameField.text === "" && !nameField.focus)
                        ? "transparent"
                        : (nameField.focus ? "#ffffff" : "white")
                    Behavior on color { ColorAnimation { duration: 200 } }
                    width: nameField.focus ? parent.width : parent.width * 0.5
                    Behavior on width { NumberAnimation { duration: 200; easing.type: Easing.InOutQuad } }
                }
            }
        }

        Button {
            id: continueButton
            text: "Continue"
            width: parent.width * 0.5
            height: 20 + window.height * 0.05
            enabled: tempName !== ""
            palette.button: "transparent"
            hoverEnabled: true
            
            anchors.horizontalCenter: parent.horizontalCenter

            background: Rectangle {
                width: continueButton.width
                height: continueButton.height
                radius: height / 2             // pill shape
                color: continueButton.hovered ? "white" : "transparent"
                border.width: 2
                border.color: "white"
                Behavior on color { ColorAnimation { duration: 1000; easing.type: Easing.InOutQuad } }
            }

            contentItem: Text {
                text: continueButton.text
                font.family: satoshi.name
                font.pixelSize: window.height * 0.025 + 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                color: continueButton.hovered ? "black" : "white"
                Behavior on color { ColorAnimation { duration: 1000; easing.type: Easing.InOutQuad } }

            }

            onClicked: {
                backend.saveUser(tempName)
                controller.continueToApp()
            }

        }
    }
}
