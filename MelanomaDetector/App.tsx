import {StatusBar} from 'expo-status-bar'
import React from 'react'
import {StyleSheet, Text, View, TouchableOpacity, Alert, ImageBackground, Image, Dimensions, Platform} from 'react-native'
import {Camera, CameraType} from 'expo-camera'
let camera: Camera
export default function App() {
  const [startCamera, setStartCamera] = React.useState(false)
  const [previewVisible, setPreviewVisible] = React.useState(false)
  const [capturedImage, setCapturedImage] = React.useState<any>(null)
  const [cameraType, setType] = React.useState(CameraType.back)
  const [flashMode, setFlashMode] = React.useState('off')

  // Screen Ratio and image padding
  const [imagePadding, setImagePadding] = React.useState(0);
  const [ratio, setRatio] = React.useState('4:3');  // default is 4:3
  const { height, width } = Dimensions.get('window');
  const screenRatio = height / width;
  const [isRatioSet, setIsRatioSet] = React.useState(false);

  const __startCamera = async () => {
    const {status} = await Camera.requestCameraPermissionsAsync()
    console.log(status)
    if (status === 'granted') {
      setCameraReady()
    } else {
      Alert.alert('Access denied')
    }
  }

  // set the camera ratio and padding.
  // this code assumes a portrait mode screen
  const prepareRatio = async () => {
    let desiredRatio = '4:3';  // Start with the system default
    // This issue only affects Android
    if (Platform.OS === 'android') {
      const ratios = await camera.getSupportedRatiosAsync();

      // Calculate the width/height of each of the supported camera ratios
      // These width/height are measured in landscape mode
      // find the ratio that is closest to the screen ratio without going over
      let distances = {};
      let realRatios = {};
      let minDistance = null;
      for (const ratio of ratios) {
        const parts = ratio.split(':');
        const realRatio = parseInt(parts[0]) / parseInt(parts[1]);
        realRatios[ratio] = realRatio;
        // ratio can't be taller than screen, so we don't want an abs()
        const distance = screenRatio - realRatio; 
        distances[ratio] = distance;
        if (minDistance == null) {
          minDistance = ratio;
        } else {
          if (distance >= 0 && distance < distances[minDistance]) {
            minDistance = ratio;
          }
        }
      }
      // set the best match
      desiredRatio = minDistance;
      //  calculate the difference between the camera width and the screen height
      const remainder = Math.floor(
        (height - realRatios[desiredRatio] * width) / 2
      );
      // set the preview padding and preview ratio
      setImagePadding(remainder);
      setRatio(desiredRatio);
      // Set a flag so we don't do this 
      // calculation each time the screen refreshes
      setIsRatioSet(true);
    }
  }
  // the camera must be loaded in order to access the supported ratios
  const setCameraReady = async() => {
    if (!isRatioSet) {
      await prepareRatio();
    }
  }
  const __takePicture = async () => {
    const photo: any = await camera.takePictureAsync()
    console.log(photo)
    setPreviewVisible(true)
    //setStartCamera(false)
    setCapturedImage(photo)
  }
  const __savePhoto = () => {}
  const __retakePicture = () => {
    setCapturedImage(null)
    setPreviewVisible(false)
    __startCamera()
  }
  const __handleFlashMode = () => {
    if (flashMode === 'on') {
      setFlashMode('off')
    } else if (flashMode === 'off') {
      setFlashMode('on')
    } else {
      setFlashMode('auto')
    }
  }
  const __switchCamera = () => {
    setType(current => (current === CameraType.back ? CameraType.front : CameraType.back));
  }
  return (
    <View style={styles.container}>
      {startCamera ? (
        <View
          style={{
            flex: 1,
            width: '100%'
          }}
        >
          {previewVisible && capturedImage ? (
            <CameraPreview photo={capturedImage} savePhoto={__savePhoto} retakePicture={__retakePicture} />
          ) : (
            <Camera
              type={cameraType}
              style={[styles.cameraPreview, {marginTop: imagePadding, marginBottom: imagePadding}]}
              onCameraReady={setCameraReady}
              ratio={ratio}
              ref={(r) => {
              camera=r;
              }}>
              <View
                style={{
                  flex: 1,
                  width: '100%',
                  backgroundColor: 'transparent',
                  flexDirection: 'row'
                }}
              >
                <View
                  style={{
                    position: 'absolute',
                    left: '5%',
                    top: '10%',
                    flexDirection: 'column',
                    justifyContent: 'space-between'
                  }}
                >
                  <TouchableOpacity
                    onPress={__handleFlashMode}
                    style={{
                      backgroundColor: flashMode === 'off' ? '#000' : '#fff',
                      height: 25,
                      width: 25
                    }}
                  >
                    <Text
                      style={{
                        fontSize: 20
                      }}
                    >
                      ⚡️
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={__switchCamera}
                    style={{
                      marginTop: 20,
                      height: 25,
                      width: 25
                    }}
                  >
                    <Text
                      style={{
                        fontSize: 20
                      }}
                    >
                      {cameraType === 'front' ? '🤳' : '📷'}
                    </Text>
                  </TouchableOpacity>
                </View>
                <View
                  style={{
                    position: 'absolute',
                    bottom: 0,
                    flexDirection: 'row',
                    flex: 1,
                    width: '100%',
                    padding: 20,
                    justifyContent: 'space-between'
                  }}
                >
                  <View
                    style={{
                      alignSelf: 'center',
                      flex: 1,
                      alignItems: 'center'
                    }}
                  >
                    <TouchableOpacity
                      onPress={__takePicture}
                      style={{
                        width: 70,
                        height: 70,
                        bottom: 0,
                        borderRadius: 50,
                        backgroundColor: '#fff'
                      }}
                    />
                  </View>
                </View>
              </View>
            </Camera>
          )}
        </View>
      ) : (
        <View
          style={{
            flex: 1,
            backgroundColor: '#fff',
            justifyContent: 'center',
            alignItems: 'center'
          }}
        >
          <TouchableOpacity
            onPress={__startCamera}
            style={{
              width: 130,
              borderRadius: 4,
              backgroundColor: '#14274e',
              flexDirection: 'row',
              justifyContent: 'center',
              alignItems: 'center',
              height: 40
            }}
          >
            <Text
              style={{
                color: '#fff',
                fontWeight: 'bold',
                textAlign: 'center'
              }}
            >
              Take picture
            </Text>
          </TouchableOpacity>
        </View>
      )}

      <StatusBar style="auto" />
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center'
  },
  information: { 
    flex: 1,
    justifyContent: 'center',
    alignContent: 'center',
    alignItems: 'center',
  },
  cameraPreview: {
    flex: 1,
  }
})

const CameraPreview = ({photo, retakePicture, savePhoto}: any) => {
  console.log('sdsfds', photo)
  return (
    <View
      style={{
        backgroundColor: 'transparent',
        flex: 1,
        width: '100%',
        height: '100%'
      }}
    >
      <ImageBackground
        source={{uri: photo && photo.uri}}
        style={{
          flex: 1
        }}
      >
        <View
          style={{
            flex: 1,
            flexDirection: 'column',
            padding: 15,
            justifyContent: 'flex-end'
          }}
        >
          <View
            style={{
              flexDirection: 'row',
              justifyContent: 'space-between'
            }}
          >
            <TouchableOpacity
              onPress={retakePicture}
              style={{
                width: 130,
                height: 40,

                alignItems: 'center',
                borderRadius: 4
              }}
            >
              <Text
                style={{
                  color: '#fff',
                  fontSize: 20
                }}
              >
                Re-take
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={savePhoto}
              style={{
                width: 130,
                height: 40,

                alignItems: 'center',
                borderRadius: 4
              }}
            >
              <Text
                style={{
                  color: '#fff',
                  fontSize: 20
                }}
              >
                save photo
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </ImageBackground>
    </View>
  )
}
