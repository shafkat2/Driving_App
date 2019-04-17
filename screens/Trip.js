import React, { Component } from 'react'
import { Dimensions, FlatList, Image, ScrollView, StyleSheet, TouchableOpacity } from 'react-native'
import rgba from 'hex-to-rgba';
import Icon from 'react-native-vector-icons';
import { CircularProgress } from 'react-native-circular-progress';
import MapView, { Marker, PROVIDER_GOOGLE } from 'react-native-maps';

var TimerMixin = require('react-timer-mixin');
var reactMixin = require('react-mixin');
import * as actions from '../Action';



import { Block, Badge, Card, Text } from '../components';
import { theme, mocks, mapStyles } from '../constants';
import { styles as blockStyles } from '../components/Block';
import { connect } from 'react-redux';


const { width } = Dimensions.get('window');

class Trip extends Component {

  componentWillMount(){
  
    
    this.setInterval(() => {

      this.props.dispatch(actions.get_usage_by_row());
      this.props.dispatch(actions.get_usage_by_id(this.props.counter));
    }, 5200);

    this.setInterval(() => {

      this.props.dispatch(actions.get_refill());
      
     
    }, 10000);
    
    
}


  static navigationOptions = ({ navigation }) => {
    const showMap = navigation.getParam('map');

    return {
      headerTitle: (
        <Text style={theme.fonts.header}>
          Current Trip
        </Text>
      ),
      headerLeft: null,
      headerRight: (
        <TouchableOpacity
          onPress={() => navigation.navigate("Trip", { map: !showMap })}
        >
          <Text
            medium
            accent={showMap}
            primary={!showMap}
            transform="uppercase"
          >
            {showMap ? 'hide map' : 'show map'}
          </Text>
        </TouchableOpacity>
      )
    }
  }

  renderChart() {
    return (
      <Card shadow style={{ paddingVertical: theme.sizes.base * 2 }}>
        <Block center>
          <CircularProgress
            size={214} // can use  with * .5 => 50%
            fill={(this.props.usage.Fuel_Information/63)*100} // percentage
            lineCap="round" // line ending style
            rotation={220}
            arcSweepAngle={280}
            width={theme.sizes.base}
            tintColor={theme.colors.primary} // gradient is not supported :(
            backgroundColor={theme.colors.gray3}
            backgroundWidth={theme.sizes.base / 2}
          >
            {() => (
              <Block center middle>
                <Text h2 medium>{Number(this.props.usage.Fuel_Information).toFixed(2)}</Text>
                <Text h3 transform="uppercase">Available Fuel</Text>
              </Block>
            )}
          </CircularProgress>
        </Block>

        <Block center>
          <Text title spacing={1} style={{ marginVertical: 8 }}>
            Current Engine Status
          </Text>
          <Text>
            <Text primary>Good </Text>
            <Text gray transform="uppercase"> Condition</Text>
          </Text>
        </Block>
      </Card>
    )
  }

  renderMap() {
    return (
      <Card style={{ padding: 0, overflow: 'hidden' }}>
        <MapView
          showsMyLocationButton={true}
          region={{  latitude: this.props.usage.Coordinates.Lat,
            longitude: this.props.usage.Coordinates.long,
            latitudeDelta: 0.06,
            longitudeDelta: 0.06,}}
          provider={PROVIDER_GOOGLE}
          customMapStyle={mapStyles}
          style={styles.map}
        >
          <Marker
            rotation={-15}
            anchor={{ x: 0.5, y: 0.5 }}
            coordinate={{ latitude: this.props.usage.Coordinates.Lat, longitude: this.props.usage.Coordinates.long }}
          >
            <Badge color={rgba(theme.colors.primary, '0.2')} size={77}>
              <TouchableOpacity activeOpacity={0.8}>
                <Badge color={rgba(theme.colors.primary, '0.2')} size={57}>
                  <Icon.MaterialCommunityIcons name="car-sports" size={57 / 2.5} color="black" />
                </Badge>
              </TouchableOpacity>
            </Badge>
          </Marker>
        </MapView>
        <TouchableOpacity
          activeOpacity={0.8}
          style={[styles.mapMyLocation, blockStyles.shadow]}
          onPress={() => alert('My Location')}
        >
          <Block center middle shadow>
            <Icon.FontAwesome
              size={16}
              name="location-arrow"
              color={theme.colors.primary}
            />
          </Block>
        </TouchableOpacity>
      </Card>
    )
  }

  renderDriveStatus = drive => {
    return (
      <TouchableOpacity activeOpacity={0.8} onPress={() => {}}>
        <Card shadow style={styles.drivingStatus}>
          <Image source={drive.icon} style={styles.drivingIcon} resizeMode="contain" />
          <Text
            title
            transform="capitalize"
            accent={this.props.usage.Battery_Status === false}
            tertiary={this.props.usage.Battery_Status === true}
            primary={drive.status === 'good'}
            height={22}
          >
            {this.props.usage.Battery_Status ? "battery on":"battery off"}
          </Text>
          
          <Text
            transform="capitalize"
            spacing={0.7}
          >
            {this.props.usage.Status ? "engine on":"engine off"}
          </Text>
        </Card>
      </TouchableOpacity>
    )
  }

  renderDriving() {
    return (
      <Block>
        <Block style={{
          marginTop: theme.sizes.base,
          marginBottom: theme.sizes.base,
          paddingHorizontal: theme.sizes.base / 3
        }}
        >
          <Text spacing={0.7} transform="uppercase">
            driving data
          </Text>
        </Block>

        <FlatList
          horizontal
          pagingEnabled
          scrollEnabled
          showsHorizontalScrollIndicator={false}
          decelerationRate={0}
          scrollEventThrottle={16}
          style={{ overflow: 'visible' }}
          data={mocks.drivingData}
          keyExtractor={(item, index) => `${item.id}`}
          renderItem={({ item }) => this.renderDriveStatus(item)}
        />

        <Block row space="between" style={{ paddginVertical: theme.sizes.base * 2 }}>
          <Block center>
            <Text h3 gray medium>55</Text>
            <Text h3 gray medium>mps</Text>
          </Block>

          <Block />

          <Block center>
            <Text h3 gray medium>978.7</Text>
            <Text h3 gray medium>mi</Text>
          </Block>
        </Block>
      </Block>
    )
  }

  renderTripButton() {
    const { navigation } = this.props;

    return (
      <Block center middle style={styles.endTrip}>
        <Badge color={rgba(theme.colors.accent, '0.1')} size={144}>
          <TouchableOpacity activeOpacity={0.8} onPress={() => navigation.navigate("Welcome")}>
            <Badge color={theme.colors.accent} size={62}>
              <Icon.FontAwesome name="square" size={62 / 2.5} color="white" />
            </Badge>
          </TouchableOpacity>
        </Badge>
      </Block>
    )
  }

  render() {
    const { navigation } = this.props;
    const showMap = navigation.getParam('map');

    return (
      <React.Fragment>
        <ScrollView style={styles.trip} showsVerticalScrollIndicator={false}>
          {showMap ? this.renderMap() : this.renderChart()}
          {this.renderDriving()}
        </ScrollView>
        { this.renderTripButton()}
      </React.Fragment>
    )
  }
}

const styles = StyleSheet.create({
  trip: {
    paddingVertical: theme.sizes.padding / 2,
    padding: theme.sizes.padding,
    backgroundColor: theme.colors.gray4,
  },
  drivingStatus: {
    marginRight: theme.sizes.base,
    width: width / 2.568,
  },
  drivingIcon: {
    height: 56,
    marginBottom: theme.sizes.base * 2,
  },
  map: {
    height: 352,
  },
  mapMyLocation: {
    position: 'absolute',
    borderRadius: 4,
    bottom: theme.sizes.base,
    left: theme.sizes.base,
    width: theme.sizes.base * 3,
    height: theme.sizes.base * 3,
    backgroundColor: theme.colors.white,
  },
  endTrip: {
    position: 'absolute',
    left: (width - 144) / 2,
    bottom: 0,
  }
})
function mapStateToProps(state){

  return{
      usage: state.usage.data,
      counter: state.counter.row,
      refill: state.refill.data
  }
}
reactMixin(Trip.prototype, TimerMixin);

export default connect(mapStateToProps)(Trip);
